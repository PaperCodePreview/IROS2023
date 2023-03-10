''' Batched REVERIE navigation environment '''

import json
import os
import numpy as np
import math
import random
import networkx as nx
from collections import defaultdict
import copy
import jsonlines
import json

import MatterSim
from numpy.random import default_rng

from vlnbert.vlnbert_init import get_tokenizer
from utils.data import load_nav_graphs, new_simulator, ImageFeaturesDB
from utils.data import angle_feature, get_all_point_angle_feature
from param import args


class EnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, connectivity_dir, scan_data_dir=None, feat_db=None, num_robots=1, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.feat_db = feat_db
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60

        self.sims = []
        for i in range(batch_size):
            sims = []
            for j in range(num_robots):
                sim = MatterSim.Simulator()
                if scan_data_dir:
                    sim.setDatasetPath(scan_data_dir)
                sim.setNavGraphPath(connectivity_dir)
                sim.setRenderingEnabled(False)
                sim.setDiscretizedViewingAngles(True)  # Set increment/decrement to 30 degree. (otherwise by radians)
                sim.setCameraResolution(self.image_w, self.image_h)
                sim.setCameraVFOV(math.radians(self.vfov))
                sim.setBatchSize(1)
                sim.initialize()
                sims.append(sim)
            self.sims.append(sims)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, scanId in enumerate(scanIds):
            for j, (viewpointId, heading) in enumerate(zip(viewpointIds[i], headings[i])):
                self.sims[i][j].newEpisode([scanId], [viewpointId], [heading], [0])

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sims in enumerate(self.sims):
            states = []
            for j, sim in enumerate(sims):
                state = sim.getState()[0]
                feature = self.feat_db.get_image_feature(state.scanId, state.location.viewpointId)
                states.append((feature, state))
            feature_states.append(states)
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, action_env in enumerate(actions):
            for j, (index, heading, elevation) in enumerate(action_env):
                self.sims[i][j].makeAction([index], [heading], [elevation])


class PlannerEnv(object):
    ''' Implements the REVERIE navigation task, using discretized viewpoints and pretrained features '''

    def __init__(
            self, instr_dir, connectivity_dir, num_robots, image_feat_size,
            angle_feat_size, start_vp_cache=None, view_db_file=None, igibson_env=None, seed=0, name=None,
            split='train', multi_endpoints=True, batch_size=1,
            memory_max_length=args.memory_max_length
    ):
        self.error_margin = 3.0
        if view_db_file is not None:
            view_db = ImageFeaturesDB(view_db_file, image_feat_size)
        else:
            view_db = None
        self.igibson_env = igibson_env
        self.num_robots = num_robots
        self.connectivity_dir = connectivity_dir
        self.angle_feat_size = angle_feat_size
        self.name = name
        self.multi_endpoints = multi_endpoints
        self.batch_size = batch_size
        self.env = EnvBatch(connectivity_dir, feat_db=view_db, batch_size=batch_size, num_robots=num_robots)
        self.split = split
        instr_data, nodes = self.construct_instrs(instr_dir)
        self.data = instr_data
        self.nodes = nodes

        # x denotes the pointer at the scan axis, y denotes the pointer at the graph axis, z the robot axis
        self.ix_x = np.arange(batch_size, dtype=int)
        self.ix_y = np.zeros(batch_size, dtype=int)
        self.ix_z = np.zeros(batch_size, dtype=int)
        self.covered = np.zeros(len(self.scans), dtype=int)

        # use different seeds in different processes to shuffle data
        self.seed = seed

        self.memory = {}
        self.memory_max_length = memory_max_length
        self.sim = new_simulator(self.connectivity_dir)
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)

        self.gt_trajs = {}
        self.buffered_state_dict = {}
        if start_vp_cache:
            with open(start_vp_cache, 'r') as f:
                self.start_vp_cache = json.load(f)
        else:
            self.start_vp_cache = {}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))

    def _get_gt_trajs(self, batch):
        gt_trajs = self.gt_trajs
        for i, scan_data in enumerate(batch):
            for j, x in enumerate(scan_data):
                if "%s_%i" % (x['instr_id'], j) not in gt_trajs:
                    gt_trajs["%s_%i" % (x['instr_id'], j)] = (self.scan_batch[i], x['path'])
        return gt_trajs

    def size(self):
        return len(self.data)

    def construct_instrs(self, instr_dir):
        self.tokenizer = get_tokenizer(args)
        scene_ids = os.listdir(instr_dir)
        self.scans = [scene_id.split('.')[0] for scene_id in scene_ids]
        assert len(self.scans) >= self.batch_size

        self._load_nav_graphs()

        data = {scan: [] for scan in self.scans}
        cand_end_vps = {scan: [] for scan in self.scans}
        for instr_file in scene_ids:
            with jsonlines.open(os.path.join(instr_dir, instr_file)) as f:
                for item in f:
                    instr_id = item['id']
                    scan = instr_file.split('.')[0]
                    end_vps = item['pos_vps']
                    instruction = item['instruction']
                    heading = item['heading']
                    old_id = item['id_old']
                    instr_encoding = self.tokenizer(instruction)
                    newitem = {
                        'instr_id': instr_id,
                        'old_id': old_id,
                        'end_vps': end_vps,
                        'instruction': instruction,
                        'instr_encoding': instr_encoding,
                        'heading': np.random.rand() * np.pi * 2,
                    }
                    cand_end_vps[scan].extend(end_vps)
                    data[scan].append(newitem)

        cand_end_vps = {key: set(value) for key, value in cand_end_vps.items()}

        clean_data = {scan: [] for scan in self.scans}
        nodes = {scan: [] for scan in self.scans}
        for i, scan in enumerate(self.scans):
            graphs = self._get_subgraph(scan, cand_end_vps)
            data_i = copy.deepcopy(data[scan])
            for j, graph in enumerate(graphs):
                nodes_tmp = set(node for node in graph.nodes)
                data_tmp = []
                for d in data_i:
                    end_vps_tmp = []
                    for vp in d['end_vps']:
                        if vp in nodes_tmp:
                            end_vps_tmp.append(vp)
                    if len(end_vps_tmp):
                        d['end_vps'] = end_vps_tmp
                        data_tmp.append(d)
                if len(data_tmp) > 3:
                    data_tmp = self.sort_graph_data(data_tmp, scan)
                    clean_data[scan].append(data_tmp)
                    nodes[scan].append(nodes_tmp)

        return clean_data, nodes

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity_old graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def get_node_position(self, scan_id, node_id):
        return self.graphs[scan_id].nodes.get(node_id)['position']

    def _get_subgraph(self, scan_id, cand_end_vps):
        graphs = []
        C = nx.connected_components(self.graphs[scan_id])
        for sub_nodes in C:
            subgraph = nx.subgraph(self.graphs[scan_id], sub_nodes)
            found = False
            for node in subgraph.nodes:
                if node in cand_end_vps[scan_id]:
                    found = True
                    break
            if found:
                graphs.append(subgraph)
        # random.shuffle(self.sub_graph_cand[scan_id])
        return graphs

    def _next_episode(self):
        """
        sample start points and end points base on distance
        """

        self.scan_batch = [self.scans[x] for x in self.ix_x]
        batch = []
        for i, scan_id in enumerate(self.scan_batch):
            data_per_scan_graph = self.data[scan_id][self.ix_y[i]][self.ix_z[i]: self.ix_z[i] + self.num_robots]
            self.ix_z[i] += self.num_robots
            available = copy.deepcopy(self.nodes[scan_id][self.ix_y[i]])

            # move cursor
            if self.ix_z[i] >= len(self.data[scan_id][self.ix_y[i]]):
                assert self.num_robots // 2 <= len(self.data[scan_id][self.ix_y[i]])
                if len(data_per_scan_graph) < self.num_robots:
                    ix_tmp = self.num_robots - len(data_per_scan_graph)
                    data_per_scan_graph += self.data[scan_id][self.ix_y[i]][:ix_tmp]
                self.ix_z[i] = 0
                self.ix_y[i] += 1
                if self.ix_y[i] >= len(self.data[scan_id]):
                    self.ix_y[i] = 0
                    self.covered[self.ix_x[i]] = 1
                    while self.ix_x[i] < len(self.scans) and self.covered[self.ix_x[i]]:
                        self.ix_x[i] += 1
                        check = copy.deepcopy(self.ix_x)
                        check[i] = -1
                        check -= self.ix_x[i]
                        while not check.all():
                            self.ix_x[i] += 1
                            check = copy.deepcopy(self.ix_x)
                            check[i] = -1
                            check -= self.ix_x[i]
                            continue
                    if self.ix_x[i] >= len(self.data):
                        self.ix_x[i] = 0

            selected = set()
            for j, d in enumerate(data_per_scan_graph):
                if self.split == 'dev' and "%s_%i" % (d['instr_id'], j) in self.start_vp_cache:
                    start_vp = self.start_vp_cache["%s_%i" % (d['instr_id'], j)]
                    d['start_vps'] = [start_vp]
                else:
                    if self.split == 'train':
                        random.shuffle(d['end_vps'])  # only read the first element, so shuffle is the same as choosing
                    candidates = set()
                    for node in available:
                        if 3 < len(self.shortest_paths[scan_id][d['end_vps'][0]][node]) < 7:
                            candidates.add(node)
                    candidates = (candidates - set(d['end_vps']))
                    try:
                        start_vp = candidates.pop()
                    except Exception:
                        try:
                            start_vp = selected.pop()
                        except Exception:
                            for node in available:
                                if 2 < len(self.shortest_paths[scan_id][d['end_vps'][0]][node]) < 7:
                                    candidates.add(node)
                            candidates = (candidates - set(d['end_vps']))
                            start_vp = candidates.pop()
                    selected.add(start_vp)
                    available.discard(start_vp)
                    d['start_vps'] = [start_vp]
                    self.start_vp_cache["%s_%i" % (d['instr_id'], j)] = start_vp
                d['path'] = [self.shortest_paths[scan_id][start_vp][end_vp] for end_vp in d['end_vps']]

            batch.append(data_per_scan_graph)

        self.batch = batch

    # sort data to get the robots closer
    def sort_graph_data(self, data_per_graph, scan_id):
        for i, data in enumerate(data_per_graph):
            distance = 9999
            min_j = -1
            for j in range(i+1, len(data_per_graph)):
                if data['end_vps'][0] == data_per_graph[j]['end_vps'][0]:
                    continue
                dis_tmp = []
                for end_vp_i in data['end_vps']:
                    for end_vp_j in data_per_graph[j]['end_vps']:
                        dis_tmp.append(len(self.shortest_paths[scan_id][end_vp_i][end_vp_j]))
                if np.mean(dis_tmp) < distance:
                    min_j = j
                    distance = np.mean(dis_tmp)
            if min_j != -1:
                data_per_graph[i+1], data_per_graph[min_j] = data_per_graph[min_j], data_per_graph[i+1]
        return data_per_graph

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.scans)
            for scan_id in self.data.keys():
                for i, graph in enumerate(self.data[scan_id]):
                    random.shuffle(self.data[scan_id][i])
                    self.data[scan_id][i] = self.sort_graph_data(self.data[scan_id][i], scan_id)
            for scan_id in self.nodes.keys():
                for j, graph in enumerate(self.nodes[scan_id]):
                    graph = list(graph)
                    random.shuffle(graph)
                    self.nodes[scan_id][j] = set(graph)
        self.gt_trajs = {}
        self.ix_x = np.arange(self.batch_size, dtype=int)
        self.ix_y = np.zeros(self.batch_size, dtype=int)
        self.ix_z = np.zeros(self.batch_size, dtype=int)
        self.covered = np.zeros(len(self.scans), dtype=int)

    # viewpointId 导航点位置，viewId 视角Id
    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        base_elevation = (viewId // 12 - 1) * math.radians(30)

        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation - base_elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "normalized_elevation": state.elevation + loc.rel_elevation,
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'position': (loc.x, loc.y, loc.z),
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'normalized_elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx', 'position']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                visual_feat = feature[ix]
                c_new['heading'] = c_new['normalized_heading'] - base_heading
                c_new['elevation'] = c_new['normalized_elevation'] - base_elevation
                angle_feat = angle_feature(c_new['heading'], c_new['elevation'], self.angle_feat_size)
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                c_new.pop('normalized_elevation')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []

        for i, scan_states in enumerate(self.env.getStates()):
            scan_obs = []
            for j, (feature, state) in enumerate(scan_states):
                item = self.batch[i][j]
                base_view_id = state.viewIndex

                # Full features
                candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
                # [visual_feature, angle_feature] for views
                feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

                view_point = '%s_%s' % (state.scanId, state.location.viewpointId)
                memory_feature = np.zeros([self.memory_max_length, 2 * args.hidden_size], dtype=np.float32)
                if view_point in self.memory:
                    for k, (key, value) in enumerate(self.memory[view_point].items()):
                        memory_feature[k] = value

                ob = {
                    'instr_id': item['instr_id'],
                    'scan': state.scanId,
                    'viewpoint': state.location.viewpointId,
                    'viewIndex': state.viewIndex,
                    'position': [state.location.x, state.location.y, state.location.z],
                    'heading': state.heading,
                    'elevation': state.elevation,
                    'feature': feature,
                    'candidate': candidate,
                    'memory_feature': memory_feature,
                    # 'obj_img_fts': obj_img_fts,
                    # 'obj_ang_fts': obj_ang_fts,
                    # 'obj_box_fts': obj_box_fts,
                    # 'obj_ids': obj_ids,
                    'navigableLocations': state.navigableLocations,
                    'instruction': item['instruction'],
                    'instr_encoding': item['instr_encoding'],
                    'teacher': self.shortest_paths[state.scanId][state.location.viewpointId][item['end_vps'][0]],
                    'gt_path': item['path'],
                    'gt_end_vps': item.get('end_vps', []),
                    # 'gt_obj_id': item['objId'],
                    # 'path_id' : item['path_id']
                }
                # RL reward. The negative distance between the state and the final state
                # There are multiple gt end viewpoints on REVERIE.
                if "%s_%i" % (ob['instr_id'], j) in self.gt_trajs:
                    min_dist = np.inf
                    for vp in item['end_vps']:
                        min_dist = min(min_dist, self.shortest_distances[ob['scan']][ob['viewpoint']][vp])
                    ob['distance'] = min_dist
                else:
                    ob['distance'] = 0

                scan_obs.append(ob)
            obs.append(scan_obs)

        return obs

    def update_memory(self, scan_viewpoint, feature, robot_id):
        if scan_viewpoint not in self.memory:
            self.memory[scan_viewpoint] = {}
        self.memory[scan_viewpoint][robot_id] = feature

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_episode(**kwargs)
        viewpointIds = []
        headings = []
        for i, scan in enumerate(self.scan_batch):
            scan_viewpointIds = [item['path'][0][0] for item in self.batch[i]]
            scan_headings = [item['heading'] for item in self.batch[i]]
            viewpointIds.append(scan_viewpointIds)
            headings.append(scan_headings)
        self.env.newEpisodes(self.scan_batch, viewpointIds, headings)
        self.gt_trajs = self._get_gt_trajs(self.batch)  # for evaluation
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0]
        near_d = self.shortest_distances[scan][near_id][goal_id]
        for item in path:
            d = self.shortest_distances[scan][item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    ############### Nav Evaluation ###############
    def _eval_item(self, scan, pred_path, gt_path):
        scores = {}
        shortest_distances = self.shortest_distances[scan]
        path = [vp[0] for vp in pred_path]
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])
        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])

        # navigation: success is to arrive to a viewpoint where the object is visible
        goal_viewpoints = {gt_path[-1]}
        assert len(goal_viewpoints) > 0

        nearest_position = self._get_nearest(scan, gt_path[-1], path)
        scores['nav_errors'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_errors'] = shortest_distances[nearest_position][gt_path[-1]]
        scores['success'] = scores['nav_errors'] < self.error_margin
        scores['oracle_success'] = scores['oracle_errors'] < self.error_margin

        # scores['success'] = float(path[-1] in goal_viewpoints)
        # scores['oracle_success'] = float(any(x in goal_viewpoints for x in path))
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)

        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        calculated = set()

        metrics = defaultdict(list)
        for item in preds:
            if item['instr_id'] in calculated:
                continue
            instr_id = item['instr_id']
            calculated.add(instr_id)
            traj = item['trajectory']
            scan, gt_trajs = self.gt_trajs[instr_id]
            best_score = None
            for gt_traj in gt_trajs:
                traj_score = self._eval_item(scan, traj, gt_traj)
                if best_score is None or best_score['spl'] < traj_score['spl']:
                    best_score = traj_score
            for k, v in best_score.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

        avg_metrics = {
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
        }

        # with open('dev_start_vp_cache.json', 'w') as f:
        #     json.dump(self.start_vp_cache, f)

        return avg_metrics, metrics
