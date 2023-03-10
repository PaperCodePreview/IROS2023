import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
import line_profiler

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from planner import Seq2SeqAgent
from utils.ops import padding_idx, print_progress

from vlnbert.graph_utils import GraphMap
from utils.ops import pad_tensors_wgrad
from models import VLNBERTDUET, Critic


class GMapObjectNavAgent(Seq2SeqAgent):

    def __init__(self, train_env, device, args, results_path=None, episode_len=20):
        super(Seq2SeqAgent, self).__init__(train_env, results_path)
        self.episode_len = episode_len
        self.args = args

        # Models
        self.vln_bert = VLNBERTDUET(self.args).to(device)
        self.critic = Critic().to(device)
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
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens = []
        batch_cand_vpids = []

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.feature_size])
                view_ang_fts.append(cc['feature'][self.args.feature_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.feature_size] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.feature_size:] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).to(self.device)
        batch_loc_fts = pad_tensors(batch_loc_fts).to(self.device)
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).to(self.device)
        batch_view_lens = torch.LongTensor(batch_view_lens).to(self.device)

        return {
            'view_img_fts': batch_view_img_fts,
            'loc_fts': batch_loc_fts, 'nav_types': batch_nav_types,
            'view_lens': batch_view_lens, 'obj_lens': 0,
            'cand_vpids': batch_cand_vpids, 'obj_ids': None,
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.get_connected_nodes():
                if gmap.graph.visited(k):
                    visited_vpids.append(k)
                else:
                    unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            if len(unvisited_vpids) != 0:
                gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
                gmap_img_embeds = torch.stack(
                    [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
                )  # cuda
            else:
                gmap_img_embeds = torch.stack(
                    [torch.zeros(self.args.hidden_size, device=self.device)], 0
                )

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i + 1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).to(self.device)
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).to(self.device)
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).to(self.device)
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).to(self.device)

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.to(self.device)

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks,
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, obj_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i],
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp],
                obs[i]['heading'], obs[i]['elevation']
            )
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).to(self.device)

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().to(self.device), nav_types == 1], 1)
        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens + obj_lens + 1),
            'vp_nav_masks': vp_nav_masks,
            'vp_obj_masks': None,
            'vp_cand_vpids': [[None] + x for x in cand_vpids],
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
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
                gt_path = ob['gt_path'][-1]
                if ob['viewpoint'] == gt_path[-1]:
                    a[i] = 0  # Stop if arrived
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            dist = self.env.shortest_distances[scan][vpid][gt_path[-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).to(self.device)

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        cnt = 0
        for i, scan_ob in enumerate(obs):
            for j, ob in enumerate(scan_ob):
                action = a_t[cnt]
                if action is not None:  # None is the <stop> action
                    traj[cnt]['path'].append(gmaps[cnt].graph.path(ob['viewpoint'], action))
                    if len(traj[cnt]['path'][-1]) == 1:
                        prev_vp = traj[cnt]['path'][-2][-1]
                    else:
                        prev_vp = traj[cnt]['path'][-1][-2]
                    viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][action]
                    heading = (viewidx % 12) * math.radians(30)
                    elevation = (viewidx // 12 - 1) * math.radians(30)
                    self.env.env.sims[i][j].newEpisode([ob['scan']], [action], [heading], [elevation])
                cnt += 1

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        num_scans, num_robots = obs.shape
        batch_size = num_scans * num_robots
        obs = obs.reshape([-1])
        self._update_scanvp_cands(obs)

        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)
            # if self.args.use_memory:
            #     for j in range((i//num_robots)*num_robots, (i//num_robots + 1)*num_robots):
            #         if j == i:
            #             gmaps[j].update_graph(ob)
            #         else:
            #             gmaps[j].update_graph(ob, visit=False)
            # else:
            #     gmaps[i].update_graph(ob)
        # Record the navigation path
        traj = []
        for i, ob in enumerate(obs):
            tra = {
                'instr_id': ob['instr_id'],
                'robot_id': i % num_robots,
                'path': [[ob['viewpoint']]],
            }
            traj.append(tra)

        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        entropys = []
        ml_loss = 0.

        for t in range(self.episode_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    if self.args.use_memory:
                        for j in range((i // num_robots) * num_robots, (i // num_robots + 1) * num_robots):
                            # update visited node
                            i_vp = obs[i]['viewpoint']
                            gmaps[j].update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    else:
                        i_vp = obs[i]['viewpoint']
                        gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp) and i_cand_vp not in gmap.locked:
                            if self.args.use_memory:
                                for k in range((i // num_robots) * num_robots, (i // num_robots + 1) * num_robots):
                                    gmaps[k].update_node_embed(i_cand_vp, pano_embeds[i, j])
                            else:
                                gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['obj_lens'],
                    pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
            })
            nav_outs = self.vln_bert('navigation', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)

            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    # update i_vp: stop and object grounding scores
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item()
                    }

            if train_ml is not None:
                # Supervised training
                nav_targets = self._teacher_action(
                    obs, nav_vpids, ended,
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None
                )
                # print(t, nav_logits, nav_targets)
                ml_loss += self.criterion(nav_logits, nav_targets)
                # print(t, 'ml_loss', ml_loss.item(), self.criterion(nav_logits, nav_targets).item())
                if self.args.fusion in ['avg', 'dynamic'] and self.args.loss_nav_3:
                    # add global and local losses
                    ml_loss += self.criterion(nav_outs['global_logits'], nav_targets)  # global
                    local_nav_targets = self._teacher_action(
                        obs, nav_inputs['vp_cand_vpids'], ended, visited_masks=None
                    )
                    ml_loss += self.criterion(nav_outs['local_logits'], local_nav_targets)  # local

            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)  # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs[
                        'gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample':  # in training
                a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                # a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.maxAction - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs.reshape([num_scans, num_robots]), traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf'), 'og': None}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))

            # new observation and update graph
            obs = np.array(self.env._get_obs())
            obs = obs.reshape([-1])
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    if self.args.use_memory:
                        for j in range((i // num_robots) * num_robots, (i // num_robots + 1) * num_robots):
                            if not ended[j]:
                                if j == i:
                                    gmaps[i].update_graph(ob)
                                else:
                                    gmaps[i].update_graph(ob, visit=False)
                    else:
                        gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss = ml_loss
            self.logs['IL_loss'].append(ml_loss.item())

        return traj

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

            if self.args.train_alg == 'imitation':
                self.feedback = 'teacher'
                self.rollout(
                    train_ml=1., train_rl=False, **kwargs
                )
            elif self.args.train_alg == 'dagger':
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, **kwargs
                    )
                self.feedback = self.args.dagger_sample
                self.rollout(train_ml=1, train_rl=False, **kwargs)
            else:
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, **kwargs
                    )
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)

            # print(self.rank, iter, self.loss)
            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            print_progress(np.mean(self.env.ix_x), len(self.env.scans))

            if self.env.covered.all():
                break

    def test(self, val_env=None, use_dropout=False, feedback='argmax'):
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        env = self.env
        if val_env:
            self.env = val_env
        self.env.reset_epoch(shuffle=False)  # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        self.loss = 0

        while True:
            for traj in self.rollout():
                self.loss = 0
                if "%s_%i" % (traj['instr_id'], traj['robot_id']) not in self.results:
                    self.results["%s_%i" % (traj['instr_id'], traj['robot_id'])] = traj['path']
            print_progress(np.mean(self.env.ix_x), len(self.env.scans))
            if self.env.covered.all():
                break
        if val_env:
            self.env = env
