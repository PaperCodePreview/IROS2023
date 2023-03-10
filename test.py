import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince

# from models.vlnbert_init import get_tokenizer

from utils.data import ImageFeaturesDB

from planner import Seq2SeqAgent
from envs.env import PlannerEnv
from param import args

dataset_dir = '/home/xyz9911/Source/igibson/datasets/nav_graphs_v1'


def build_dataset():

    dataset_class = PlannerEnv

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes

    feat_db_file = os.path.join(dataset_dir, 'features', 'resnet-152')

    train_env = dataset_class(
        '/home/xyz9911/Source/igibson/datasets/nav_graphs_v1/path',
        '/home/xyz9911/Source/igibson/datasets/nav_graphs_v1/connectivity_old', num_robots=2, view_db_file=feat_db_file, batch_size=1
    )


    # # val_env_names = ['val_train_seen']
    # val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']
    #
    # if args.submit:
    #     val_env_names.append('test')
    #
    # val_envs = {}
    # for split in val_env_names:
    #     val_instr_data = construct_instrs(
    #         args.anno_dir, args.dataset, [split],
    #         tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
    #     )
    #     val_env = dataset_class(
    #         feat_db, obj_db, val_instr_data, args.connectivity_dir, obj2vps, batch_size=args.batch_size,
    #         angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
    #         sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
    #         max_objects=None, multi_endpoints=False, multi_startpoints=False,
    #     )  # evaluation using all objects
    #     val_envs[split] = val_env

    return train_env


def train(train_env, val_envs=None, eval_first=False):
    default_gpu = True

    # if default_gpu:
    #     with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
    #         json.dump(vars(args), outf, indent=4)
    #     writer = SummaryWriter(log_dir=args.log_dir)
    #     record_file = os.path.join(args.log_dir, 'train.txt')
    #     write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = Seq2SeqAgent
    listner = agent_class(train_env)

    # resume file
    start_iter = 0

    # first evaluation
    if eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        # if default_gpu:
        #     write_to_record_file(loss_str, record_file)

    start = time.time()
    # if default_gpu:
    #     write_to_record_file(
    #         '\nListener training starts, start iteration: %s' % str(start_iter), record_file
    #     )

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state": ""}}

    for idx in range(start_iter, start_iter + args.iters, 1):
        listner.logs = defaultdict(list)
        interval = 1
        iter = idx + interval

        # Train for log_every interval
        jdx_length = len(range(interval // 2))

            # Train with GT data
        listner.env = train_env
        listner.train(1, feedback=args.feedback)

            # if default_gpu:
            #     print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        # if default_gpu:
        #     # Log the training stats to tensorboard
        #     total = max(sum(listner.logs['total']), 1)  # RL: total valid actions for all examples in the batch
        #     length = max(len(listner.logs['critic_loss']), 1)  # RL: total (max length) in the batch
        #     critic_loss = sum(listner.logs['critic_loss']) / total
        #     policy_loss = sum(listner.logs['policy_loss']) / total
        #     OG_loss = sum(listner.logs['OG_loss']) / max(len(listner.logs['OG_loss']), 1)
        #     IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        #     entropy = sum(listner.logs['entropy']) / total
            # writer.add_scalar("loss/critic", critic_loss, idx)
            # writer.add_scalar("policy_entropy", entropy, idx)
            # writer.add_scalar("loss/OG_loss", OG_loss, idx)
            # writer.add_scalar("loss/IL_loss", IL_loss, idx)
            # writer.add_scalar("total_actions", total, idx)
            # writer.add_scalar("max_length", length, idx)
            # write_to_record_file(
            #     "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, OG_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
            #         total, length, entropy, IL_loss, OG_loss, policy_loss, critic_loss),
            #     record_file
            # )

        # Run validation
        loss_str = "iter {}".format(iter)
        # for env_name, env in val_envs.items():
        #     listner.env = env
        #
        #     # Get validation distance from goal under test evaluation conditions
        #     listner.test(use_dropout=False, feedback='argmax', iters=None)
        #     preds = listner.get_results()
        #     preds = merge_dist_results(all_gather(preds))

        #     if default_gpu:
        #         score_summary, _ = env.eval_metrics(preds)
        #         loss_str += ", %s " % env_name
        #         for metric, val in score_summary.items():
        #             loss_str += ', %s: %.2f' % (metric, val)
        #             writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)
        #
        #         # select model by spl
        #         if env_name in best_val:
        #             if score_summary['spl'] >= best_val[env_name]['spl']:
        #                 best_val[env_name]['spl'] = score_summary['spl']
        #                 best_val[env_name]['sr'] = score_summary['sr']
        #                 best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
        #                 listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
        #
        # if default_gpu:
        #     listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))
        #
        #     write_to_record_file(
        #         ('%s (%d %d%%) %s' % (
        #         timeSince(start, float(iter) / args.iters), iter, float(iter) / args.iters * 100, loss_str)),
        #         record_file
        #     )
        #     write_to_record_file("BEST RESULT TILL NOW", record_file)
        #     for env_name in best_val:
        #         write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)
        #

# def valid(args, train_env, val_envs, rank=-1):
#     default_gpu = is_default_gpu(args)
#
#     agent_class = GMapObjectNavAgent
#     agent = agent_class(args, train_env, rank=rank)
#
#     if args.resume_file is not None:
#         print("Loaded the listener model at iter %d from %s" % (
#             agent.load(args.resume_file), args.resume_file))
#
#     if default_gpu:
#         with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
#             json.dump(vars(args), outf, indent=4)
#         record_file = os.path.join(args.log_dir, 'valid.txt')
#         write_to_record_file(str(args) + '\n\n', record_file)
#
#     for env_name, env in val_envs.items():
#         prefix = 'submit' if args.detailed_output is False else 'detail'
#         output_file = os.path.join(args.pred_dir, "%s_%s_%s.json" % (
#             prefix, env_name, args.fusion))
#         if os.path.exists(output_file):
#             continue
#         agent.logs = defaultdict(list)
#         agent.env = env
#
#         iters = None
#         start_time = time.time()
#         agent.test(
#             use_dropout=False, feedback='argmax', iters=iters)
#         print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
#         preds = agent.get_results(detailed_output=args.detailed_output)
#         preds = merge_dist_results(all_gather(preds))
#
#         if default_gpu:
#             if 'test' not in env_name:
#                 score_summary, _ = env.eval_metrics(preds)
#                 loss_str = "Env name: %s" % env_name
#                 for metric, val in score_summary.items():
#                     loss_str += ', %s: %.2f' % (metric, val)
#                 write_to_record_file(loss_str + '\n', record_file)
#
#             if args.submit:
#                 json.dump(
#                     preds, open(output_file, 'w'),
#                     sort_keys=True, indent=4, separators=(',', ': ')
#                 )


def main():

    train_env = build_dataset()

    train(train_env)
    # valid(args, train_env, val_envs, rank=rank)
    # valid_viz(args, train_env, val_envs, rank=rank)


if __name__ == '__main__':
    main()
