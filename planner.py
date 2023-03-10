# R2R-EnvDrop, 2019, haotan@cs.unc.edu
# Modified in Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import json
import os
import sys
import numpy as np
import random
import math
import time
from numpy.random import default_rng

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from envs.env import PlannerEnv
from utils.data import angle_feature
from utils.ops import length2mask
from utils.misc import ndtw_initialize
from utils.ops import padding_idx, print_progress
import models
from collections import defaultdict


class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = []  # For learning agents

    def write_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name + "Agent"]

    def test(self, val_env=None, **kwargs):
        env = self.env
        if val_env:
            self.env = val_env
            self.ndtw_criterion = ndtw_initialize(val_env.scans, val_env.connectivity_dir)
        self.env.reset_epoch(shuffle=False)  # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        self.loss = 0

        while True:
            for traj in self.rollout(**kwargs):
                self.loss = 0
                if "%s_%i" % (traj['instr_id'], traj['robot_id']) not in self.results:
                    self.results["%s_%i" % (traj['instr_id'], traj['robot_id'])] = traj['path']
            print_progress(np.mean(self.env.ix_x), len(self.env.scans))
            if self.env.covered.all():
                break
        if val_env:
            self.env = env
            self.ndtw_criterion = ndtw_initialize(self.env.scans, self.env.connectivity_dir)


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    def __init__(self, train_env, device, args, results_path=None, episode_len=20):
        super(Seq2SeqAgent, self).__init__(train_env, results_path)
        self.episode_len = episode_len
        self.args = args

        # Models
        self.vln_bert = models.VLNBERT().to(device)
        self.critic = models.Critic().to(device)
        self.models = (self.vln_bert, self.critic)

        self.scanvp_cands = {}

        self.device = device

        # Optimizers
        self.vln_bert_optimizer = self.args.optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        self.critic_optimizer = self.args.optimizer(self.critic.parameters(), lr=self.args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, size_average=False)
        self.ndtw_criterion = ndtw_initialize(train_env.scans, train_env.connectivity_dir)

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding'].data['input_ids']) for ob in obs]

        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding'].data['input_ids']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().to(self.device)
        mask = torch.from_numpy(mask).to(self.device)
        token_type_ids = torch.zeros_like(mask).to(self.device)

        return {
            'mode': 'language', 'sentence': seq_tensor, 'attention_mask': mask, 'lang_mask': mask,
            'token_type_ids': token_type_ids
        }

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), self.args.views, self.args.feature_size + self.args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']  # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).to(self.device)

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.args.feature_size + self.args.angle_feat_size),
                                  dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = cc['feature']

        return torch.from_numpy(candidate_feat).to(self.device), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), self.args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = angle_feature(ob['heading'], ob['elevation'], self.args.angle_feat_size)
        input_a_t = torch.from_numpy(input_a_t).to(self.device)
        # f_t = self._feature_variable(obs)      # Pano image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        memory_feat = np.stack([ob['memory_feature'] for ob in obs])
        memory_feat = torch.from_numpy(memory_feat).to(self.device)

        return input_a_t, candidate_feat, candidate_leng, memory_feat

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if len(ob['teacher']) > 1:
                    for k, candidate in enumerate(ob['candidate']):
                        if candidate['viewpointId'] == ob['teacher'][1]:  # Next view point
                            a[i] = k
                            break
                else:  # Stop here
                    assert ob['teacher'][0] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).to(self.device)

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    def make_equiv_action(self, a_t, obs, language_feature, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """

        cnt = 0
        for i, scan_ob in enumerate(obs):
            for j, ob in enumerate(scan_ob):
                action = a_t[i][j]
                if action != -1:  # None is the <stop> action
                    prev_vp = ob['viewpoint']
                    view_point = ob['candidate'][action]['viewpointId']
                    # if len(traj[i]['path'][-1]) == 1:
                    #     prev_vp = traj[i]['path'][-2][-1]
                    # else:
                    #     prev_vp = traj[i]['path'][-1][-2]
                    viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][view_point]
                    heading = (viewidx % 12) * math.radians(30)
                    elevation = (viewidx // 12 - 1) * math.radians(30)
                    self.env.env.sims[i][j].newEpisode([ob['scan']], [view_point], [heading], [elevation])

                    state = language_feature[cnt, 0, :]
                    sentence_feature = torch.sum(language_feature[cnt, 1:, :], dim=0)
                    feature = torch.cat([state, sentence_feature], dim=-1)
                    self.env.update_memory('%s_%s' % (ob['scan'], prev_vp), feature.cpu().numpy(), robot_id=j)

                    traj[cnt]['path'].append((view_point, heading, elevation))

                cnt += 1

        return traj

    def rollout(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        num_scans, num_robots = obs.shape

        batch_size = num_scans * num_robots
        obs = obs.reshape([-1])

        self._update_scanvp_cands(obs)

        ''' Language BERT '''
        # Language input
        language_inputs = self._language_variable(obs)

        h_t, language_features = self.vln_bert(**language_inputs)

        # Record starting point
        traj = []
        for i, ob in enumerate(obs):
            tra = {
                'instr_id': ob['instr_id'],
                'robot_id': i % num_robots,
                'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            }
            traj.append(tra)

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):  # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            max_ndtw_score = -999
            for gt_path in ob['gt_path']:
                ndtw_score_tmp = self.ndtw_criterion[ob['scan']](path_act, gt_path, metric='ndtw')
                if ndtw_score_tmp > max_ndtw_score:
                    max_ndtw_score = ndtw_score_tmp
            last_ndtw[i] = max_ndtw_score

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        for t in range(self.episode_len):
            input_a_t, candidate_feat, candidate_leng, memory_feat = self.get_input_feat(obs)

            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            language_features = torch.cat((h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1)

            visual_temp_mask = (length2mask(candidate_leng, device=self.device) == 0).long()
            visual_attention_mask = torch.cat((language_inputs['lang_mask'], visual_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode': 'visual',
                             'sentence': language_features,
                             'attention_mask': visual_attention_mask,
                             'lang_mask': language_inputs['lang_mask'],
                             'vis_mask': visual_temp_mask,
                             'token_type_ids': language_inputs['token_type_ids'],
                             'action_feats': input_a_t,
                             'memory_feats': memory_feat,
                             # 'pano_feats':         f_t,
                             'cand_feats': candidate_feat}
            h_t, logit = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = length2mask(candidate_leng, device=self.device)
            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            target = self._teacher_action(obs, ended)
            ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)  # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i] - 1) or next_id == self.args.ignoreid or ended[
                    i]:  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # Make action and get the new state
            traj = self.make_equiv_action(cpu_a_t.reshape([num_scans, num_robots]),
                                          obs.reshape([num_scans, num_robots]), language_features.detach(), traj)
            obs = np.array(self.env._get_obs())
            obs = obs.reshape([-1])

            self._update_scanvp_cands(obs)

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    max_ndtw_score = -999
                    for gt_path in ob['gt_path']:
                        ndtw_score_tmp = self.ndtw_criterion[ob['scan']](path_act, gt_path, metric='ndtw')
                        if ndtw_score_tmp > max_ndtw_score:
                            max_ndtw_score = ndtw_score_tmp
                    ndtw_score[i] = max_ndtw_score
                    # ndtw_score[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:  # If the action now is end
                            if dist[i] < 3.0:  # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:  # Incorrect
                                reward[i] = -2.0
                        else:  # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:  # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                reward[i] = ndtw_reward
                                # raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i] - last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            # Last action in A2C
            input_a_t, candidate_feat, candidate_leng, memory_feat = self.get_input_feat(obs)

            language_features = torch.cat((h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1)

            visual_temp_mask = (length2mask(candidate_leng, device=self.device) == 0).long()
            visual_attention_mask = torch.cat((language_inputs['lang_mask'], visual_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode': 'visual',
                             'sentence': language_features,
                             'attention_mask': visual_attention_mask,
                             'lang_mask': language_inputs['lang_mask'],
                             'vis_mask': visual_temp_mask,
                             'token_type_ids': language_inputs['token_type_ids'],
                             'action_feats': input_a_t,
                             'memory_feats': memory_feat,
                             # 'pano_feats':         f_t,
                             'cand_feats': candidate_feat}
            last_h_, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()  # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:  # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).to(self.device)
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).to(self.device)
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)  # This argument is useless.

        return traj

    def test(self, val_env=None, use_dropout=False, feedback='argmax'):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super(Seq2SeqAgent, self).test(val_env=val_env)

    def train(self, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.env.reset_epoch(shuffle=True)  # If iters is not none, shuffle the env batch
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        while True:

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            print_progress(np.mean(self.env.ix_x), len(self.env.scans))

            if self.env.covered.all():
                break

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            # if self.args.loadOptim:
            #     optimizer.load_state_dict(states[name]['optimizer'])

        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
