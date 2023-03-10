import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from mushroom_rl.core import Agent
from mushroom_rl.utils.parameters import to_parameter
from gym import spaces

from utils.ops import obs_as_tensor, partial
from utils.data import parse_dataset, minibatch_generator, compute_J
from models import MlpExtractor, GAT


class PPO(Agent):
    """
    Proximal Policy Optimization algorithm.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.

    """

    def __init__(self, mdp_info, policy, optimizer,
                 n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff=0.01, vf_coef: float = 0.5, max_grad_norm=0.5, logger=None):
        """
        Constructor.

        Args:
            policy (TorchPolicy): torch policy to be learned by the algorithm
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            n_epochs_policy ([int, Parameter]): number of policy updates for every dataset;
            batch_size ([int, Parameter]): size of minibatches for every optimization step
            eps_ppo ([float, Parameter]): value for probability ratio clipping;
            lam ([float, Parameter], 1.): lambda coefficient used by generalized
                advantage estimation;
            ent_coeff ([float, Parameter], 1.): coefficient for the entropy regularization term;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._n_epochs_policy = to_parameter(n_epochs_policy)
        self._batch_size = to_parameter(batch_size)
        self._eps_ppo = to_parameter(eps_ppo)

        self._optimizer = optimizer['class'](policy.parameters(), **optimizer['params'])

        self._lambda = to_parameter(lam)
        self._ent_coeff = to_parameter(ent_coeff)

        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self._iter = 1
        self._logger = logger

        super().__init__(mdp_info, policy, None)

    def fit(self, dataset):
        x, u, r, old_log_p, v, vn, mask, absorbing, last = parse_dataset(dataset)
        x = {key: value.astype(np.float32) for key, value in x.items()}
        # u = u.astype(np.float32)
        # r = np.expand_dims(r.astype(np.float32), -1)

        # obs = obs_as_tensor(x, self.policy.device)
        # act = obs_as_tensor(u, self.policy.device)
        r = obs_as_tensor(r, self.policy.device)
        mask = obs_as_tensor(mask, self.policy.device)
        # obs_n = obs_as_tensor(xn, self.policy.device)

        # compute gae
        gen_adv = torch.zeros_like(v, device=self.policy.device)
        for rev_k in range(len(v)):
            k = len(v) - rev_k - 1
            if last[k].any() or rev_k == 0:
                if rev_k == 0:
                    gen_adv[k] = r[k] - v[k]
                else:
                    for i in range(len(last[k])):
                        if last[k][i]:
                            gen_adv[k][i] = r[k][i] - v[k][i]
                if not absorbing[k].any():
                    gen_adv[k] += self.mdp_info.gamma * vn[k]
                else:
                    for i in range(len(absorbing[k])):
                        if not absorbing[k][i]:
                            gen_adv[k][i] += self.mdp_info.gamma * vn[k][i]
            else:
                gen_adv[k] = r[k] + self.mdp_info.gamma * vn[k] - v[k] + self.mdp_info.gamma * self._lambda() * gen_adv[k + 1]

        v_target, adv = gen_adv + v, gen_adv

        adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)

        pl, vl, el, kl = self._update_policy(x, u, adv, old_log_p, v_target, mask)

        # Print fit information
        self._log_info(dataset, pl, vl, el, kl)
        self._iter += 1

    # @torch.no_grad()
    # def predict_values(self, state):
    #     obs = obs_as_tensor(state, self.policy.device)
    #     values = self.policy.predict_values(obs)
    #     return values

    @torch.no_grad()
    def draw_action(self, state):
        """
        Return the action to execute in the given state. It is the action
        returned by the policy or the action set by the algorithm (e.g. in the
        case of SARSA).

        Args:
            state (np.ndarray): the state where the agent is.

        Returns:
            The action to be executed.

        """
        obs = obs_as_tensor(state, self.policy.device)
        actions, values, log_probs = self.policy(obs)

        # Clip the actions to avoid out of bound error

        return actions, values, log_probs

    def _update_policy(self, obs, act, adv, old_log_p, v_target, mask):
        pg_loss = []
        v_loss = []
        e_loss = []
        approx_kl = []
        for epoch in range(self._n_epochs_policy()):
            for obs_i, act_i, adv_i, old_log_p_i, v_target_i, mask_i in minibatch_generator(
                    self._batch_size(), obs, act, adv, old_log_p, v_target, mask):
                self._optimizer.zero_grad()

                obs_i = obs_as_tensor(obs_i, self.policy.device)
                # act_i = obs_as_tensor(act_i, self.policy.device)
                # adv_i = obs_as_tensor(adv_i, self.policy.device).squeeze(-1)
                # old_log_p_i = obs_as_tensor(old_log_p_i, self.policy.device)
                # v_target_i = obs_as_tensor(v_target_i, self.policy.device)

                # adv_i = adv_i.squeeze(-1)
                adv_i *= mask_i

                values, log_prob, entropy = self.policy.evaluate_actions(obs_i, act_i)

                prob_ratio = torch.exp(log_prob - old_log_p_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(),
                                            1 + self._eps_ppo.get_value())

                value_loss = self.vf_coef*F.mse_loss(values*mask_i, v_target_i*mask_i, reduction='sum')/torch.sum(mask_i)
                entropy_loss = -self._ent_coeff() * entropy
                policy_loss = -torch.sum(torch.min(prob_ratio * adv_i, clipped_ratio * adv_i))/torch.sum(mask_i)

                loss = policy_loss + value_loss + entropy_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self._optimizer.step()

                pg_loss.append(policy_loss.item())
                v_loss.append(value_loss.item())
                e_loss.append(entropy_loss.item())
                with torch.no_grad():
                    approx_kl_div = torch.mean((torch.exp(log_prob) - 1) - log_prob).cpu().numpy()
                approx_kl.append(approx_kl_div)

        return np.mean(pg_loss), np.mean(v_loss), np.mean(e_loss), np.mean(approx_kl)

    @torch.no_grad()
    def _log_info(self, dataset, pl, vl, el, kl):
        if self._logger:
            # logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old_pol_dist))
            avg_rwd = np.mean(compute_J(dataset))
            msg = "Iteration {}:\n\t\t\t\trewards {} pi_loss{} vf_loss {} e_loss{} kl{}".format(
                self._iter, avg_rwd, pl, vl, el, kl)

            self._logger.info(msg)
            self._logger.weak_line()


class GaussianTorchPolicy(nn.Module):
    """
    Torch policy implementing a Gaussian policy with trainable standard
    deviation. The standard deviation is not state-dependent.

    """

    def __init__(self, features_extractor_class, observation_space, action_space, device, net_arch=None,
                 activation_fn=nn.Tanh,
                 std_0=1., gnn_feature_dim=128, **features_extractor_kwargs):
        """
        Constructor.

        Args:
            network (object): the network class used to implement the mean
                regressor;
            input_shape (tuple): the shape of the state space;
            output_shape (tuple): the shape of the action space;
            std_0 (float, 1.): initial standard deviation;
            params (dict): parameters used by the network constructor.

        """
        super().__init__()
        self._action_dim = self.get_action_dim(action_space)
        self.action_space = action_space
        self.observation_space = observation_space
        self.features_extractor_kwargs = features_extractor_kwargs
        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor._features_dim

        self.gnn = GAT(self.features_extractor._features_dim, gnn_feature_dim, gnn_feature_dim, 4)

        self.features_dim = self.features_extractor._features_dim + gnn_feature_dim

        self.device = device

        if net_arch is None:
            net_arch = [dict(pi=[128, 128], vf=[128, 128])]
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )

        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, self._action_dim)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        self._init()

        log_sigma_init = (torch.ones(self._action_dim) * np.log(std_0)).float()

        self._log_sigma = nn.Parameter(log_sigma_init)

    def _init(self):
        module_gains = {
            self.features_extractor: np.sqrt(2),
            self.mlp_extractor: np.sqrt(2),
            self.action_net: 0.01,
            self.value_net: 1,
        }
        for module, gain in module_gains.items():
            module.apply(partial(self.init_weights, gain=gain))

    def process_features(self, obs, is_numpy=False):
        if is_numpy:
            obs = obs_as_tensor(obs, self.device)
        adj_mat = obs.pop('adj_mat')
        if len(adj_mat.size()) > 2:
            batch_size, robot_num = adj_mat.size()[:2]
            for key, value in obs.items():
                obs[key] = value.view((batch_size * robot_num,) + value.size()[2:])
            features = self.features_extractor(obs)
            features = features.view(batch_size, robot_num, -1)
        else:
            features = self.features_extractor(obs)
        gnn_features = self.gnn(features, adj_mat.unsqueeze(dim=-1))
        features = torch.cat([features, gnn_features], dim=-1)
        return features

    def forward(self, obs):
        features = self.process_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        mu = self.action_net(latent_pi)
        sigma = torch.diag(torch.exp(2 * self._log_sigma))
        distribution = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values.squeeze(-1), log_probs

    def evaluate_actions(self, obs, actions):
        features = self.process_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        mu = self.action_net(latent_pi)
        sigma = torch.diag(torch.exp(2 * self._log_sigma))
        distribution = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values.squeeze(-1), log_prob, self.entropy_t(obs)

    def entropy_t(self, state=None):
        return self._action_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self._log_sigma)

    def get_action_dim(self, action_space: spaces.Space) -> int:
        """
        Get the dimension of the action space.

        :param action_space:
        :return:
        """
        if isinstance(action_space, spaces.Box):
            return int(np.prod(action_space.shape))
        elif isinstance(action_space, spaces.Discrete):
            # Action is an int
            return 1
        elif isinstance(action_space, spaces.MultiDiscrete):
            # Number of discrete actions
            return int(len(action_space.nvec))
        elif isinstance(action_space, spaces.MultiBinary):
            # Number of binary actions
            return int(action_space.n)
        else:
            raise NotImplementedError(f"{action_space} action space is not supported")

    # def draw_action(self, obs):
    #     with torch.no_grad():
    #         features = self.process_features(obs, True)
    #         latent_pi = self.mlp_extractor.forward_actor(features)
    #         mu = self.action_net(latent_pi)
    #         sigma = torch.diag(torch.exp(2 * self._log_sigma))
    #         distribution = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
    #         a = distribution.sample()
    #
    #     actions = a.detach().cpu().numpy()
    #
    #     # Clip the actions to avoid out of bound error
    #     clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
    #
    #     return clipped_actions

    def entropy(self, state=None):
        s = obs_as_tensor(state, self.device) if state is not None else None
        return self.entropy_t(s).detach().cpu().numpy().item()

    def predict_values(self, obs):
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features = self.process_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf).cpu().numpy()

    def reset(self):
        pass

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
