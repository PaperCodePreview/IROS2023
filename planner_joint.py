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

from navigation.algorithm_v2.ppo import PPO, GaussianTorchPolicy
from navigation.algorithm_v2.models import CombinedExtractor

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from utils.data import angle_feature
from utils.ops import length2mask
from utils.misc import ndtw_initialize
from utils.ops import padding_idx, print_progress
import models
import param
from param import args
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

    def test(self, val_env=None, use_controller=False, **kwargs):
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
            for traj in self.rollout(use_controller=use_controller, val=True, **kwargs):
                self.loss = 0
                if "%i_%i" % (traj['instr_id'], traj['robot_id']) not in self.results:
                    self.results["%i_%i" % (traj['instr_id'], traj['robot_id'])] = traj['path']
            print_progress(np.mean(self.env.ix_x), len(self.env.scans))
            if self.env.covered.all():
                break
        if val_env:
            self.env = env
            self.ndtw_criterion = ndtw_initialize(self.env.scans, self.env.connectivity_dir)


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
        'left': [[0], [-1], [0]],  # left
        'right': [[0], [1], [0]],  # right
        'up': [[0], [0], [1]],  # up
        'down': [[0], [0], [-1]],  # down
        'forward': [[1], [0], [0]],  # forward
        '<end>': [[0], [0], [0]],  # <end>
        '<start>': [[0], [0], [0]],  # <start>
        '<ignore>': [[0], [0], [0]]  # <ignore>
    }

    def __init__(self, train_env, planner_device, controller_device=None, controller_params=None, policy_params=None,
                 igibson_env=None, results_path=None, controller_reset=True, episode_max_len=20, logger=None):
        super(Seq2SeqAgent, self).__init__(train_env, results_path)
        self.episode_max_len = episode_max_len

        self._controller_total_steps = 0
        self._controller_current_steps = 0
        self._controller_fit_steps = None
        self._fit_condition = None
        self._controller_success = np.ones(train_env.num_robots)
        self._controller_reset = controller_reset

        if igibson_env is not None:
            assert igibson_env.num_robots == train_env.num_robots

        self.rollout_buffer = []
        self.mdp = igibson_env

        self.device = planner_device

        # Models
        self.vln_bert = models.VLNBERT().to(planner_device)
        self.critic = models.Critic().to(planner_device)
        self.models = (self.vln_bert, self.critic)
        if self.mdp is not None:
            self.policy = GaussianTorchPolicy(CombinedExtractor,
                                              self.mdp.info.observation_space,
                                              self.mdp.info.action_space,
                                              device=controller_device,
                                              **policy_params).to(controller_device)
            self.controller = PPO(self.mdp.info, self.policy, **controller_params, logger=logger)
            self.controller.set_logger(logger=logger)

        self.scanvp_cands = {}

        # Optimizers
        self.vln_bert_optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        self.ndtw_criterion = ndtw_initialize(train_env.scans, train_env.connectivity_dir)

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

        # others
        with open('reachable.json', 'r') as f:
            self.reachable_points = json.load(f)

    def load_controller(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)  # 从本地读取
        self.policy.load_state_dict(checkpoint)

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
        features = np.empty((len(obs), args.views, args.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']  # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).to(self.device)

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), args.feature_size + args.angle_feat_size),
                                  dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = cc['feature']

        return torch.from_numpy(candidate_feat).to(self.device), candidate_leng

    def _prepare_controller_condition(self, controller_fit_steps):
        if controller_fit_steps is not None:
            self._controller_fit_steps = controller_fit_steps
            self._fit_condition = lambda: self._controller_current_steps >= self._controller_fit_steps

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = angle_feature(ob['heading'], ob['elevation'], args.angle_feat_size)
        input_a_t = torch.from_numpy(input_a_t).to(self.device)
        # f_t = self._feature_variable(obs)      # Pano image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        memory_feat = np.stack([ob['memory_feature'] for ob in obs])
        memory_feat = torch.from_numpy(memory_feat).to(self.device)

        return input_a_t, candidate_feat, candidate_leng, memory_feat

    def _teacher_action(self, obs, ended, in_progress):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i] or in_progress[i]:  # Just ignore this index
                a[i] = args.ignoreid
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

    def _reset_controller(self, obs):
        initial_pos = [ob['position'] for ob in obs]
        self._state = self.mdp.reset((initial_pos, initial_pos)).copy()

        self._actions = None
        self._values = None
        self._log_probs = None
        self._clipped_actions = None
        self._mask = None
        self._controller_total_steps = 0

        self._success_count = np.zeros(self.env.num_robots)
        self._time_consumed = np.zeros(self.env.num_robots)
        self._reward_count = np.zeros(self.env.num_robots)

    def _controller_step(self, new_targets=[]):
        """
        Single step.

        Args:
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the
            agent, the reward obtained, the reached state, the absorbing flag
            of the reached state and the last step flag.

        """
        if self._actions is None:  # initialising
            self._state, _, _, _ = self.mdp.step({'actions': None, 'new_targets': new_targets})
            new_targets = []
            actions, values, log_probs = self.controller.draw_action(self._state)
            clipped_actions = np.clip(actions.cpu().numpy(), self.controller.policy.action_space.low,
                                      self.controller.policy.action_space.high)
            mask = np.ones(clipped_actions.shape[0])
        else:
            actions, clipped_actions, values, log_probs = self._actions, self._clipped_actions, self._values, self._log_probs
            mask = self._mask
        clipped_actions *= np.expand_dims(mask, axis=1).repeat(self.controller.policy._action_dim, axis=1)
        union_actions = {'actions': clipped_actions, 'new_targets': new_targets}

        next_state, reward, done, info = self.mdp.step(union_actions)
        self._mask = np.logical_not(done)

        last = done
        state = self._state

        next_actions, next_values, next_log_probs = self.controller.draw_action(next_state)
        next_clipped_actions = np.clip(next_actions.cpu().numpy(), self.controller.policy.action_space.low,
                                       self.controller.policy.action_space.high)

        self._state = next_state
        self._actions = next_actions
        self._clipped_actions = next_clipped_actions
        self._values = next_values
        self._log_probs = next_log_probs

        self._controller_success = info['success']

        return state, actions, reward, log_probs, values, next_values, mask, done, last

    def _controller_loop(self, new_targets):
        last = np.zeros(self.env.num_robots)
        cnt = 0
        while last.sum() < self.env.num_robots:
            sample = self._controller_step(new_targets)
            new_targets = []
            last_new = sample[-1]

            # do not count to buffer when init, because initial and target is the same
            if self._controller_total_steps > 0:
                self.rollout_buffer.append(sample)

            self._controller_total_steps += 1
            self._controller_current_steps += 1

            if self._fit_condition():
                self.controller.fit(self.rollout_buffer)
                self._controller_current_steps = 0

                self.rollout_buffer = []

            if cnt != 0:
                if (last_new ^ last).sum() > 0:
                    last = last_new
                    break
            last = last_new
            cnt += 1
        return last

    def make_equiv_action(self, a_t, obs, language_feature, traj=None, use_controller=False):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """

        view_points, headings, elevations, prev_vps = [], [], [], []
        new_targets = []

        cnt = 0
        for i, scan_ob in enumerate(obs):
            for j, ob in enumerate(scan_ob):
                if self.episode_len[cnt] <= self.episode_max_len:
                    action = a_t[i][j]
                    if action != -1 and action != -2:  # None is the <stop> action
                        prev_vp = ob['viewpoint']
                        view_point = ob['candidate'][action]['viewpointId']
                        viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][view_point]
                        heading = (viewidx % 12) * math.radians(30)
                        elevation = (viewidx // 12 - 1) * math.radians(30)

                        if use_controller:
                            target = self.env.get_node_position(ob['scan'], view_point)
                            new_targets.append({'robot_idx': j, 'target': target})
                            # if self._controller_success[j] or self._controller_reset:
                            #     new_targets.append(
                            #         {'robot_idx': j, 'target': target,
                            #          'reset_robot': 1 - self._controller_success[j]})
                        view_points.append(view_point)
                        headings.append(heading)
                        elevations.append(elevation)
                        prev_vps.append(prev_vp)
                cnt += 1

        if use_controller:
            done = self._controller_loop(new_targets)
            in_progress = np.logical_not(done)
        else:
            in_progress = np.zeros(a_t.shape[0] * a_t.shape[1])

        cnt = 0
        idx = 0
        for i, scan_ob in enumerate(obs):
            for j, ob in enumerate(scan_ob):
                action = a_t[i][j]
                if action != -1 and action != -2:
                    if use_controller:
                        if not self._controller_success[j]:
                            if not self._controller_reset:
                                a_t[i][j] = -3  # cut
                                cnt += 1
                                idx += 1
                                continue
                            else:
                                self.mdp.reset_robot_position(cnt)

                    self.env.env.sims[i][j].newEpisode([ob['scan']], [view_points[idx]], [headings[idx]],
                                                       [elevations[idx]])
                    state = language_feature[cnt, 0, :]
                    sentence_feature = torch.sum(language_feature[cnt, 1:, :], dim=0)
                    feature = torch.cat([state, sentence_feature], dim=-1)
                    self.env.update_memory('%s_%s' % (ob['scan'], prev_vps[idx]), feature.cpu().numpy(), robot_id=j)

                    traj[cnt]['path'].append((view_points[idx], headings[idx], elevations[idx]))
                    self.episode_len[cnt] += 1
                    idx += 1
                cnt += 1

        if use_controller:
            for i in range(self.env.num_robots):
                if a_t[0][i] == -1 or a_t[0][i] == -3:
                    self.mdp.remove_robot(i)

        return traj, a_t.reshape([-1]), in_progress

    def rollout(self, train_ml=None, train_rl=True, reset=True, use_controller=False, controller_fit_steps=None,
                val=False):
        """
        :param use_controller: Whether to use controller during planner rollout
        :param controller_fit_steps: The controller rollout buffer length
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if use_controller:
            assert self.env.batch_size == 1
            self._prepare_controller_condition(controller_fit_steps)

        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        num_scans, num_robots = obs.shape

        batch_size = num_scans * num_robots
        obs = obs.reshape([-1])

        if use_controller:
            self._reset_controller(obs)

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
            max_ndtw_score = -99999
            for gt_path in ob['gt_path']:
                ndtw_score_tmp = self.ndtw_criterion[ob['scan']](path_act, gt_path, metric='ndtw')
                if ndtw_score_tmp > max_ndtw_score:
                    max_ndtw_score = ndtw_score_tmp
            last_ndtw[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env
        cuts = []
        in_progress = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        cpu_a_ts = []
        ml_loss = 0.

        self.episode_len = np.zeros(batch_size)

        while True:
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
            target = self._teacher_action(obs, ended, in_progress)
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
                if next_id == (candidate_leng[i] - 1) or next_id == args.ignoreid or ended[
                    i] or in_progress[i]:  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1
                    if in_progress[i]:
                        cpu_a_t[i] = -2

            # Make action and get the new state
            traj, cpu_a_t, in_progress = self.make_equiv_action(cpu_a_t.reshape([num_scans, num_robots]),
                                                                obs.reshape([num_scans, num_robots]),
                                                                language_features.detach(), traj, use_controller)
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
                    max_ndtw_score = -99999
                    for gt_path in ob['gt_path']:
                        ndtw_score_tmp = self.ndtw_criterion[ob['scan']](path_act, gt_path, metric='ndtw')
                        if ndtw_score_tmp > max_ndtw_score:
                            max_ndtw_score = ndtw_score_tmp
                    ndtw_score[i] = max_ndtw_score

                    if ended[i] or in_progress[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        if action_idx == -3:
                            reward[i] = 0.0
                            mask[i] = 0.0
                        # Target reward
                        elif action_idx == -1:  # If the action now is end
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
                cpu_a_ts.append(cpu_a_t)
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            cut = np.logical_or(self.episode_len > self.episode_max_len, cpu_a_t == -3)
            ended[:] = np.logical_or(ended, (np.logical_or(cpu_a_t == -1, cut)))
            cuts.append(cut)

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            # Last action in A2C
            # input_a_t, candidate_feat, candidate_leng, memory_feat = self.get_input_feat(obs)
            #
            # language_features = torch.cat((h_t.unsqueeze(1), language_features[:, 1:, :]), dim=1)
            #
            # visual_temp_mask = (length2mask(candidate_leng) == 0).long()
            # visual_attention_mask = torch.cat((language_inputs['lang_mask'], visual_temp_mask), dim=-1)
            #
            # self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            # ''' Visual BERT '''
            # visual_inputs = {'mode': 'visual',
            #                  'sentence': language_features,
            #                  'attention_mask': visual_attention_mask,
            #                  'lang_mask': language_inputs['lang_mask'],
            #                  'vis_mask': visual_temp_mask,
            #                  'token_type_ids': language_inputs['token_type_ids'],
            #                  'action_feats': input_a_t,
            #                  'memory_feats': memory_feat,
            #                  # 'pano_feats':         f_t,
            #                  'cand_feats': candidate_feat}
            # last_h_, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            # last_value__ = self.critic(last_h_).detach()  # The value esti of the last state, remove the grad for safety
            # discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            # for i in range(batch_size):
            #     if not ended[i]:  # If the action is not ended, use the value function as the last reward
            #         discount_reward[i] = last_value__[i]

            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero

            gamma_matrix = (np.array(cpu_a_ts) != -2)  # consider the actions in progress when calculating rewards
            gamma_matrix = gamma_matrix * (args.gamma - 1) + 1

            length = len(rewards)
            total = 0
            for t in range(length - 1, -1, -1):
                v_ = self.critic(hidden_states[t])
                discount_reward = discount_reward * gamma_matrix[t] + rewards[t]  # If it ended, the reward will be 0
                for i in range(batch_size):  # if action is cut, calculate the last state value
                    if cuts[t][i]:
                        discount_reward[i] = v_[i]
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).to(self.device)
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).to(self.device)
                a_ = (r_ - v_).detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['policy_loss'].append((-policy_log_probs[t] * a_ * mask_).sum().item())
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len.sum())  # This argument is useless.

        return traj

    def test(self, val_env=None, use_dropout=False, feedback='argmax', use_controller=False):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super(Seq2SeqAgent, self).test(val_env=val_env, use_controller=use_controller)

    def train(self, feedback='teacher', use_controller=False, controller_fit_steps=1024, **kwargs):
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
                self.rollout(train_ml=args.teacher_weight, train_rl=False, use_controller=use_controller,
                             controller_fit_steps=controller_fit_steps, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.ml_weight, train_rl=False, use_controller=use_controller,
                                 controller_fit_steps=controller_fit_steps, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, use_controller=use_controller,
                             controller_fit_steps=controller_fit_steps, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            print_progress(np.mean(self.env.ix_x), len(self.env.scans))

            if self.env.covered.all():
                break

            # if args.aug is None:
            #     print_progress(iter, n_iters + 1, prefix='Progress:', suffix='Complete', bar_length=50)

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
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])

        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
