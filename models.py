import numpy as np

# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
from gnn.gatv2 import GraphAttentionV2Layer
from itertools import zip_longest
import collections

from vlnbert.vlnbert_init import get_vlnbert_models


class VLNBERTDUET(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.featdropout)

    def forward(self, mode, batch):
        batch = collections.defaultdict(lambda: None, batch)

        if mode == 'language':
            txt_embeds = self.vln_bert(mode, batch)
            return txt_embeds

        elif mode == 'panorama':
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            pano_embeds, pano_masks = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks

        elif mode == 'navigation':
            outs = self.vln_bert(mode, batch)
            return outs

        else:
            raise NotImplementedError('wrong mode: %s' % mode)


class VLNBERT(nn.Module):
    def __init__(self, feature_size=2048 + 128):
        super(VLNBERT, self).__init__()
        print('\nInitalizing the VLN-BERT model ...')

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.vln_bert.config.directions = 4  # a preset random number

        hidden_size = self.vln_bert.config.hidden_size
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        self.action_state_project = nn.Sequential(
            nn.Linear(hidden_size + args.angle_feat_size, hidden_size), nn.Tanh())
        self.action_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=args.featdropout)
        self.img_projection = nn.Linear(feature_size, hidden_size, bias=True)
        self.cand_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.vis_lang_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        self.state_proj = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.state_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.memory_query = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.memory_key = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.memory_value = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, mode, sentence, token_type_ids=None,
                attention_mask=None, lang_mask=None, vis_mask=None,
                position_ids=None, action_feats=None, pano_feats=None, cand_feats=None, memory_feats=None):

        if mode == 'language':
            init_state, encoded_sentence = self.vln_bert(mode, sentence, attention_mask=attention_mask,
                                                         lang_mask=lang_mask, )

            return init_state, encoded_sentence

        elif mode == 'visual':
            if args.use_memory:
                state_query = torch.cat((sentence[:, 0, :], torch.sum(sentence[:, 1:, :], dim=1)), 1)
                state_query = self.memory_query(state_query).unsqueeze(1)
                batch_size, memory_size, hidden_size = memory_feats.size()
                hidden_size //= 2
                memory_key = self.memory_key(memory_feats)
                attn = torch.bmm(state_query, memory_key.transpose(1, 2))
                attn = F.softmax(attn.view(-1, memory_size), dim=1).view(batch_size, -1, memory_size)
                memory_value = self.memory_value(memory_feats[:, :, :hidden_size])
                sentence[:, 0, :] += torch.bmm(attn, memory_value).squeeze(1)

            state_action_embed = torch.cat((sentence[:, 0, :], action_feats), 1)
            state_with_action = self.action_state_project(state_action_embed)
            state_with_action = self.action_LayerNorm(state_with_action)
            state_feats = torch.cat((state_with_action.unsqueeze(1), sentence[:, 1:, :]), dim=1)

            cand_feats[..., :-args.angle_feat_size] = self.drop_env(cand_feats[..., :-args.angle_feat_size])

            # logit is the attention scores over the candidate features
            h_t, logit, attended_language, attended_visual = self.vln_bert(mode, state_feats,
                                                                           attention_mask=attention_mask,
                                                                           lang_mask=lang_mask, vis_mask=vis_mask,
                                                                           img_feats=cand_feats)

            # update agent's state, unify history, language and vision by elementwise product
            vis_lang_feat = self.vis_lang_LayerNorm(attended_language * attended_visual)
            state_output = torch.cat((h_t, vis_lang_feat), dim=-1)
            state_proj = self.state_proj(state_output)
            state_proj = self.state_LayerNorm(state_proj)

            return state_proj, logit

        else:
            ModuleNotFoundError


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class CombinedExtractor(nn.Module):
    def __init__(self, observation_space, task_feature_size=128, feature_size=512, dropout=0.6):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__()
        extractors = {}

        total_concat_size = 0
        task_feature_size = task_feature_size
        feature_size = feature_size
        for key, subspace in observation_space.spaces.items():
            if key == "task_obs":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], task_feature_size), nn.ReLU())
                total_concat_size += task_feature_size
            elif key == "rgb":
                n_input_channels = subspace.shape[2]  # channel last
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=(3, 3), stride=2, padding=1),
                    nn.ELU(),
                    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
                    nn.ELU(),
                    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
                    nn.ELU(),
                    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
                    nn.ELU(),
                    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
                    nn.ELU(),
                    nn.Flatten(),
                )
                test_tensor = torch.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
                with torch.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.Dropout(dropout), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
                total_concat_size += feature_size
            else:
                raise ValueError("Unknown observation key: %s" % key)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        unsqueeze = True
        if len(observations['rgb'].size()) > 3:
            unsqueeze = False

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if unsqueeze:
                observations[key] = observations[key].unsqueeze(dim=0)
            if key == "rgb":
                observations[key] = observations[key].permute((0, 3, 1, 2))
                # observations[key] = observations[key].permute((2, 0, 1))
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        if unsqueeze:
            return torch.cat(encoded_tensor_list, dim=1).squeeze(dim=0)
        else:
            return torch.cat(encoded_tensor_list, dim=1)


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous feature extractor (i.e. a CNN) or directly
    the observations (if no feature extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.
    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
            self,
            feature_dim,
            net_arch,
            activation_fn,
    ):
        super().__init__()
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared, layer))  # add linear of size layer
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for pi_layer_size, vf_layer_size in zip_longest(policy_only_layers, value_only_layers):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net)
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

    def forward(self, features):
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features):
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features):
        return self.value_net(self.shared_net(features))


class GAT(nn.Module):
    """
    ## Graph Attention Network (GAT)

    This graph attention network has two [graph attention layers](index.html).
    """

    def __init__(self, in_features: int, out_features: int, n_hidden: int, n_heads: int, dropout: float = 0.6,
                 share_weights: bool = False):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        """
        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionV2Layer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout,
                                            share_weights=share_weights)
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        # Final graph attention layer where we average the heads
        self.output = GraphAttentionV2Layer(n_hidden, out_features, 1, is_concat=False, dropout=dropout,
                                            share_weights=share_weights)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer (without activation) for logits
        return self.output(x, adj_mat)


class Network(nn.Module):
    n_features = 512

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(n_input, 32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        dummy_obs = torch.zeros(1, *input_shape)
        conv_out_size = np.prod(self.feat_extract(dummy_obs).shape)

        self.fully_connect = nn.Sequential(
            init_(nn.Linear(conv_out_size, self.n_features)),
            nn.ReLU(),
            init_(nn.Linear(self.n_features, self.n_features)),
            nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.output_layer = init_(nn.Linear(self.n_features, n_output))

    def forward(self, state, action=None):
        q = self.feat_extract(state.float() / 255.)
        q = self.fully_connect(q.view(state.shape[0], -1))
        q = self.output_layer(q)

        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))
            return q_acted
