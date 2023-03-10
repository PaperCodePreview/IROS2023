import os
import json
import jsonlines
import numpy as np

import lmdb
import msgpack
import msgpack_numpy

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


def load_instr_datasets(anno_dir, dataset, splits, tokenizer):
    data = []
    for split in splits:
        if "/" not in split:  # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, 'REVERIE_%s_enc.json' % split)
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, 'REVERIE_%s_enc_xlmr.json' % split)
            else:
                raise NotImplementedError('unspported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)
        else:  # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            if split.endswith('json'):
                with open(split) as f:
                    new_data = json.load(f)
            elif split.endswith('jsonl'):
                # reuse pretrain aug format
                with jsonlines.open(split) as f:
                    new_data = []
                    for item in f:
                        objid = item['instr_id'].split('_')[1]
                        new_data.append({
                            'scan': item['scan'],
                            'id': '%s_%d' % (item['instr_id'], len(new_data)),
                            'instructions': [''],
                            'instr_encodings': [item['instr_encoding']],
                            'path_id': '%s_%d' % (item['instr_id'], len(new_data)),
                            'objId': objid,
                            'path': item['path'],
                            'heading': np.random.rand() * np.pi * 2,
                            'end_vps': item['pos_vps'],
                        })
            else:
                raise NotImplementedError('unsupported aug data format %s' % split)
        # Join
        data += new_data
    return data


def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            if 'objId' in item:
                new_item['instr_id'] = '%s_%s_%d' % (str(item['path_id']), str(item['objId']), j)
            else:
                new_item['path_id'] = item['id']
                new_item['instr_id'] = '%s_%d' % (item['id'], j)
                new_item['objId'] = None
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data


def load_obj2vps(bbox_file):
    obj2vps = {}
    bbox_data = json.load(open(bbox_file))
    for scanvp, value in bbox_data.items():
        scan, vp = scanvp.split('_')
        # for all visible objects at that viewpoint
        for objid, objinfo in value.items():
            if objinfo['visible_pos']:
                # if such object not already in the dict
                obj2vps.setdefault(scan + '_' + objid, [])
                obj2vps[scan + '_' + objid].append(vp)
    return obj2vps
