import os
import json
import time
import numpy as np
from collections import defaultdict
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince

from navigation.algorithm_v2.envs.ma_env_base import iGibson
from planner_joint import Seq2SeqAgent
from planner_duet_joint import GMapObjectNavAgent
from envs.env import PlannerEnv
from param import args

from mushroom_rl.core import Logger

dataset_dir = '/home/xyz9911/Source/igibson/datasets/igibson_matterport_v2'

device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def build_dataset(num_robots):
    dataset_class = PlannerEnv
    if args.model == 'Vanilla-VLNBERT':
        feat_db_file_train = os.path.join(dataset_dir, 'train', 'features_resnet152')
        feat_db_file_unseen = os.path.join(dataset_dir, 'val_unseen', 'features_resnet152')
        feat_db_file_test = os.path.join(dataset_dir, 'test', 'features_resnet152')
        image_feat_size = args.feature_size
        angle_feat_size = args.angle_feat_size
    elif args.model == 'DUET':
        feat_db_file_train = os.path.join(dataset_dir, 'train', 'features_vit')
        feat_db_file_unseen = os.path.join(dataset_dir, 'val_unseen', 'features_vit')
        feat_db_file_test = os.path.join(dataset_dir, 'test', 'features_vit')
        image_feat_size = 768
        angle_feat_size = 4

    train_env = dataset_class(
        os.path.join(dataset_dir, 'train', 'path'),
        os.path.join(dataset_dir, 'connectivity'), image_feat_size=image_feat_size, angle_feat_size=angle_feat_size,
        num_robots=num_robots,
        view_db_file=feat_db_file_train, split='train', name='train', batch_size=1
    )

    val_seen_env = dataset_class(
        os.path.join(dataset_dir, 'val_seen', 'path'),
        os.path.join(dataset_dir, 'connectivity'), image_feat_size=image_feat_size, angle_feat_size=angle_feat_size,
        num_robots=num_robots,
        start_vp_cache='val_seen_start_vps.json', view_db_file=feat_db_file_train, split='dev', name='val_seen',
        batch_size=1
    )

    val_unseen_env = dataset_class(
        os.path.join(dataset_dir, 'val_unseen', 'path'),
        os.path.join(dataset_dir, 'connectivity'), image_feat_size=image_feat_size, angle_feat_size=angle_feat_size,
        num_robots=num_robots,
        start_vp_cache='val_unseen_start_vps.json', view_db_file=feat_db_file_unseen, split='dev', name='val_unseen',
        batch_size=1
    )

    # train_env = dataset_class(
    #     os.path.join(dataset_dir, 'test', 'path'),
    #     os.path.join(dataset_dir, 'test', 'connectivity'), image_feat_size=image_feat_size, angle_feat_size=angle_feat_size, num_robots=num_robots,
    #     view_db_file=feat_db_file_test, split='train', name='train', batch_size=2
    # )
    #
    # val_env = dataset_class(
    #     os.path.join(dataset_dir, 'test', 'path'),
    #     os.path.join(dataset_dir, 'test', 'connectivity'), image_feat_size=image_feat_size, angle_feat_size=angle_feat_size, num_robots=num_robots,
    #     view_db_file=feat_db_file_test, split='train', name='val', batch_size=2
    # )

    # test = dataset_class(
    #     os.path.join(dataset_dir, 'test', 'path'),
    #     os.path.join(dataset_dir, 'test', 'connectivity'), image_feat_size=image_feat_size, angle_feat_size=angle_feat_size, num_robots=num_robots,
    #     view_db_file=feat_db_file_test, split='test', name='test'
    # )

    return train_env, val_seen_env, val_unseen_env


def train(listner, train_env, val_env_seen, val_env_unseen, igibson_env=None, n_steps_per_fit=None, eval_first=False):
    writer = SummaryWriter(log_dir=args.log_dir)
    record_file = os.path.join(args.log_dir, 'train.txt')
    write_to_record_file(str(args) + '\n\n', record_file)

    # resume file
    start_iter = 0

    # first evaluation
    if eval_first:
        loss_str = "validation before training"
        # Get validation distance from goal under test evaluation conditions
        listner.test(val_env=val_env_seen, use_dropout=False, feedback='argmax')
        preds = listner.get_results()
        score_summary, _ = val_env_seen.eval_metrics(preds)
        loss_str += ", %s " % val_env_seen.name
        for metric, val in score_summary.items():
            loss_str += ', %s: %.2f' % (metric, val)
        write_to_record_file(loss_str, record_file)

        loss_str = "validation before training"
        listner.test(val_env=val_env_unseen, use_dropout=False, feedback='argmax')
        preds = listner.get_results()
        score_summary, _ = val_env_unseen.eval_metrics(preds)
        loss_str += ", %s " % val_env_unseen.name
        for metric, val in score_summary.items():
            loss_str += ', %s: %.2f' % (metric, val)
        write_to_record_file(loss_str, record_file)

    start = time.time()
    write_to_record_file(
        '\nListener training starts, start iteration: %s' % str(start_iter), record_file
    )

    best_val = {val_env_unseen.name: {"spl": 0., "sr": 0., "state": ""},
                val_env_seen.name: {"spl": 0., "sr": 0., "state": ""}}

    for idx in range(start_iter, start_iter + args.iters, 1):
        listner.logs = defaultdict(list)
        iter = idx

        # Train with GT data
        listner.env = train_env
        # listner.train(feedback=args.feedback)
        listner.train(feedback=args.feedback, controller_fit_steps=n_steps_per_fit, use_controller=True)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)  # RL: total valid actions for all examples in the batch
        length = max(len(listner.logs['critic_loss']), 1)  # RL: total (max length) in the batch
        critic_loss = sum(listner.logs['critic_loss']) / total
        policy_loss = sum(listner.logs['policy_loss']) / total
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        write_to_record_file(
            "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                total, length, entropy, IL_loss, policy_loss, critic_loss),
            record_file
        )

        # Run validation
        loss_str = "iter {}".format(idx)

        # Get validation distance from goal under test evaluation conditions
        listner.test(val_env=val_env_seen, use_dropout=False, feedback='argmax')
        # listner.test(val_env=val_env, feedback='argmax', controller_fit_steps=n_steps_per_fit, use_controller=True)
        preds = listner.get_results()

        score_summary, _ = val_env_seen.eval_metrics(preds)
        loss_str += ", %s " % val_env_seen.name
        for metric, val in score_summary.items():
            loss_str += ', %s: %.2f' % (metric, val)
            writer.add_scalar('%s/%s' % (metric, val_env_seen.name), score_summary[metric], idx)

        write_to_record_file(
            ('%s (%d %d%%) %s' % (
                timeSince(start), iter, float(iter) / args.iters * 100, loss_str)),
            record_file
        )

        # select model by spl
        env_name = val_env_seen.name
        if env_name in best_val:
            if score_summary['sr'] >= best_val[env_name]['sr']:
                best_val[env_name]['spl'] = score_summary['spl']
                best_val[env_name]['sr'] = score_summary['sr']
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                listner.save(idx, os.path.join(args.ckpt_dir, "%s_step%i_best_%s" % (args.model, idx, env_name)))

        loss_str = "iter {}".format(idx)
        # Get validation distance from goal under test evaluation conditions
        listner.test(val_env=val_env_unseen, use_dropout=False, feedback='argmax')
        # listner.test(val_env=val_env, feedback='argmax', controller_fit_steps=n_steps_per_fit, use_controller=True)
        preds = listner.get_results()

        score_summary, _ = val_env_unseen.eval_metrics(preds)
        loss_str += ", %s " % val_env_unseen.name
        for metric, val in score_summary.items():
            loss_str += ', %s: %.2f' % (metric, val)
            writer.add_scalar('%s/%s' % (metric, val_env_unseen.name), score_summary[metric], idx)

        write_to_record_file(
            ('%s (%d %d%%) %s' % (
                timeSince(start), iter, float(iter) / args.iters * 100, loss_str)),
            record_file
        )

        # select model by spl
        env_name = val_env_unseen.name
        if env_name in best_val:
            if score_summary['sr'] >= best_val[env_name]['sr']:
                best_val[env_name]['spl'] = score_summary['spl']
                best_val[env_name]['sr'] = score_summary['sr']
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                listner.save(idx, os.path.join(args.ckpt_dir, "%s_step%i_best_%s" % (args.model, idx, env_name)))

        listner.save(idx, os.path.join(args.ckpt_dir, "%s_latest" % args.model))

        write_to_record_file("BEST RESULT TILL NOW", record_file)
        for env_name in best_val:
            write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def main():
    num_robots = 4
    # args.model = 'DUET'
    train_env, val_seen_env, val_unseen_env = build_dataset(num_robots)

    default_config = os.path.join('config.yaml')
    mdp = iGibson(config_file=default_config, task_random=False, debug_gui=False, num_robots=num_robots)
    n_steps_per_fit = 512
    policy_params = dict(
        std_0=1.,
    )
    alg_params = dict(optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         n_epochs_policy=4,
                         batch_size=64,
                         eps_ppo=.2,
                         lam=.95)

    logger = Logger()
    if args.model == 'Vanilla-VLNBERT':
        listner = Seq2SeqAgent(train_env, device_0, device_1, igibson_env=mdp, controller_params=alg_params,
                              policy_params=policy_params, logger=logger)
        listner.load_controller('SingleGaussianTorchPolicy.pt')
    elif args.model == 'DUET':
        agent_class = GMapObjectNavAgent
        args.feature_size = 768
        args.angle_feat_size = 4

    train(listner, train_env, val_seen_env, val_unseen_env, igibson_env=mdp, n_steps_per_fit=n_steps_per_fit)


if __name__ == '__main__':
    main()
