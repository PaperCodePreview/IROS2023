import os
import json
import jsonlines
import networkx as nx
import math
import numpy as np

import lmdb
import msgpack
import msgpack_numpy
import torch

msgpack_numpy.patch()


class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}
        self.env = lmdb.open(self.img_ft_file, readonly=True)

    def __del__(self):
        self.env.close()

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with self.env.begin() as txn:
                ft = msgpack.unpackb(txn.get(key.encode('ascii')))
            ft = ft[:, :self.image_feat_size].astype(np.float32)
            self._feature_store[key] = ft
        return ft


def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity_old graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2 \
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2 \
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            # C = sorted(nx.connected_components(G), key=len, reverse=True)
            graphs[scan] = G
    return graphs


def new_simulator(connectivity_dir, scan_data_dir=None, width=640, height=480, vfov=60):
    import MatterSim

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(width, height)
    sim.setCameraVFOV(math.radians(vfov))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    return sim


def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)


def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = baseViewId * math.radians(30)

    for ix in range(36):
        heading = (ix % 12) * math.radians(30) - base_heading
        elevation = (ix // 12 - 1) * math.radians(30)
        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature


def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]


def parse_dataset(dataset, features=None):
    """
    Split the dataset in its different components and return them.

    Args:
        dataset (list): the dataset to parse;
        features (object, None): features to apply to the states.

    Returns:
        The np.ndarray of state, action, reward, next_state, absorbing flag and
        last step flag. Features are applied to ``state`` and ``next_state``,
        when provided.

    """
    assert len(dataset) > 0

    keys = dataset[0][0].keys()

    state = {key: np.ones((len(dataset),) + dataset[0][0][key].shape) for key in keys}
    action = torch.stack([dataset[i][1] for i in range(len(dataset))])
    reward = np.ones((len(dataset),) + dataset[0][2].shape)
    old_log_p = torch.stack([dataset[i][3] for i in range(len(dataset))])
    value = torch.stack([dataset[i][4] for i in range(len(dataset))])
    next_value = torch.stack([dataset[i][5] for i in range(len(dataset))])
    mask = np.ones((len(dataset),) + dataset[0][6].shape)
    absorbing = np.ones((len(dataset),) + dataset[0][7].shape)
    last = np.ones((len(dataset),) + dataset[0][8].shape)

    if features is not None:
        for i in range(len(dataset)):
            for key in keys:
                state[key][i, ...] = features(dataset[i][0][key])
            reward[i] = dataset[i][2]
            mask[i] = dataset[i][6]
            absorbing[i] = dataset[i][7]
            last[i] = dataset[i][8]
    else:
        for i in range(len(dataset)):
            for key in keys:
                state[key][i, ...] = dataset[i][0][key]
            reward[i] = dataset[i][2]
            mask[i] = dataset[i][6]
            absorbing[i] = dataset[i][7]
            last[i] = dataset[i][8]

    return state, action, np.array(reward), old_log_p, value, next_value, np.array(mask), np.array(absorbing), np.array(
        last)


def minibatch_number(size, batch_size):
    """
    Function to retrieve the number of batches, given a batch sizes.

    Args:
        size (int): size of the dataset;
        batch_size (int): size of the batches.

    Returns:
        The number of minibatches in the dataset.

    """
    return int(np.ceil(size / batch_size))


def minibatch_generator(batch_size, *dataset):
    """
    Generator that creates a minibatch from the full dataset.

    Args:
        batch_size (int): the maximum size of each minibatch;
        dataset: the dataset to be splitted.

    Returns:
        The current minibatch.

    """
    size = len(dataset[1])
    num_batches = minibatch_number(size, batch_size)
    indexes = np.arange(0, size, 1)
    np.random.shuffle(indexes)
    batches = [(i * batch_size, min(size, (i + 1) * batch_size))
               for i in range(0, num_batches)]

    for (batch_start, batch_end) in batches:
        batch = []
        for i in range(len(dataset)):
            if isinstance(dataset[i], dict):
                batch.append({key: v[indexes[batch_start:batch_end]] for key, v in dataset[i].items()})
            else:
                batch.append(dataset[i][indexes[batch_start:batch_end]])
        yield batch


def compute_J(dataset, gamma=1.):
    """
    Compute the cumulative discounted reward of each episode in the dataset.

    Args:
        dataset (list): the dataset to consider;
        gamma (float, 1.): discount factor.

    Returns:
        The cumulative discounted reward of each episode in the dataset.

    """
    js = list()

    j = 0.
    episode_steps = 0
    for i in range(len(dataset)):
        j += gamma ** episode_steps * np.mean(dataset[i][2])
        episode_steps += 1
        if np.sum(dataset[i][-1]) == len(dataset[i][-1]) or i == len(dataset) - 1:
            js.append(j)
            j = 0.
            episode_steps = 0

    if len(js) == 0:
        return [0.]
    return js
