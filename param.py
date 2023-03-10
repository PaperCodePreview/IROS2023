import argparse
import os
import torch


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--test_only', type=int, default=0, help='fast mode for testing')

        self.parser.add_argument('--iters', type=int, default=300, help='training iterations')
        self.parser.add_argument('--name', type=str, default='Vanilla-VLNBERT', help='experiment id')
        self.parser.add_argument('--train', type=str, default='listener')
        self.parser.add_argument('--description', type=str, default='no description\n')
        self.parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')
        self.parser.add_argument('--model', choices=['Vanilla-VLNBERT', 'DUET'], default='Vanilla-VLNBERT')
        self.parser.add_argument('--use_memory', type=bool, default=False)

        # duet
        self.parser.add_argument('--enc_full_graph', default=False, action='store_true')
        self.parser.add_argument('--fusion', choices=['global', 'local', 'avg', 'dynamic'], default='dynamic')
        self.parser.add_argument('--dagger_sample', choices=['sample', 'expl_sample', 'argmax'], default='sample')
        self.parser.add_argument('--expl_max_ratio', type=float, default=0.6)
        self.parser.add_argument('--loss_nav_3', action='store_true', default=False)
        self.parser.add_argument('--entropy_loss_weight', type=float, default=0.01)
        self.parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
        self.parser.add_argument('--fix_pano_embedding', action='store_true', default=False)
        self.parser.add_argument('--fix_local_branch', action='store_true', default=False)
        self.parser.add_argument('--num_l_layers', type=int, default=9)
        self.parser.add_argument('--num_pano_layers', type=int, default=2)
        self.parser.add_argument('--num_x_layers', type=int, default=4)
        self.parser.add_argument('--graph_sprels', action='store_true', default=False)
        self.parser.add_argument('--train_alg',
                                 choices=['imitation', 'dagger', 'a3c', 'reinforce'],
                                 default='dagger'
                                 )

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=15, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=8)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim", action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--load", default=None, help='path of the trained model')

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.3)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)

        # Dropout Param
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # Submision configuration
        self.parser.add_argument("--submit", type=int, default=0)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')  # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--feedback', type=str, default='sample',
                                 help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                                 help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)
        self.parser.add_argument('--memory_max_length', type=int, default=10)

        self.parser.add_argument('--hidden_size', type=int, default=768)
        self.parser.add_argument('--log_every', type=int, default=100)
        self.parser.add_argument('--dev_split', type=int, default=0.1)
        self.parser.add_argument('--ckpt_dir', type=str, default='/data/checkpoints')

        # Model hyper params:
        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=128)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            print("Optimizer: Using AdamW")
            self.args.optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False


param = Param()
args = param.args

args.name = args.model
args.log_dir = 'snap/%s' % args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
