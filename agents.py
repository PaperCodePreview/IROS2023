from tqdm import tqdm
import numpy as np


class Controller:
    """
    Implements the functions to run a generic algorithm.

    """

    def __init__(self, controller, env, callbacks_fit=None, callback_step=None,
                 preprocessors=None):
        """
        Constructor.

        Args:
            agent (Agent): the agent moving according to a policy;
            mdp (Environment): the environment in which the agent moves;
            callbacks_fit (list): list of callbacks to execute at the end of
                each fit;
            callback_step (Callback): callback to execute after each step;
            preprocessors (list): list of state preprocessors to be
                applied to state variables before feeding them to the
                agent.

        """
        self.controller = controller
        self.mdp = env
        self.callbacks_fit = callbacks_fit if callbacks_fit is not None else list()
        self.callback_step = callback_step if callback_step is not None else lambda x: None
        self._preprocessors = preprocessors if preprocessors is not None else list()

        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0
        self._episode_steps = None
        self._n_episodes = None
        self._n_steps_per_fit = None
        self._n_episodes_per_fit = None

    def train(self, n_steps=None, n_episodes=None, n_steps_per_fit=None,
              n_episodes_per_fit=None, render=False, quiet=False):
        """
        This function moves the agent in the environment and fits the policy
        using the collected samples. The agent can be moved for a given number
        of steps or a given number of episodes and, independently from this
        choice, the policy can be fitted after a given number of steps or a
        given number of episodes. By default, the environment is reset.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            n_steps_per_fit (int, None): number of steps between each fit of the
                policy;
            n_episodes_per_fit (int, None): number of episodes between each fit
                of the policy;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        assert (n_episodes_per_fit is not None and n_steps_per_fit is None) \
               or (n_episodes_per_fit is None and n_steps_per_fit is not None)

        self._n_steps_per_fit = n_steps_per_fit
        self._n_episodes_per_fit = n_episodes_per_fit

        if n_steps_per_fit is not None:
            fit_condition = \
                lambda: self._current_steps_counter >= self._n_steps_per_fit
        else:
            fit_condition = lambda: self._current_episodes_counter \
                                    >= self._n_episodes_per_fit

        self._run(n_steps, n_episodes, fit_condition, render, quiet)

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=False):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from
        a set of initial states for the whole episode. By default, the
        environment is reset.

        Args:
            initial_states (np.ndarray, None): the starting states of each
                episode;
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        fit_condition = lambda: False

        return self._run(n_steps, n_episodes, fit_condition, render, quiet,
                         initial_states)

    def _run(self, n_steps, n_episodes, fit_condition, render, quiet,
             initial_states=None):
        assert n_episodes is not None and n_steps is None and initial_states is None \
               or n_episodes is None and n_steps is not None and initial_states is None \
               or n_episodes is None and n_steps is None and initial_states is not None

        self._n_episodes = len(
            initial_states) if initial_states is not None else n_episodes

        if n_steps is not None:
            move_condition = \
                lambda: self._total_steps_counter < n_steps

            steps_progress_bar = tqdm(total=n_steps,
                                      dynamic_ncols=True, disable=quiet,
                                      leave=False)
            episodes_progress_bar = tqdm(disable=True)
        else:
            move_condition = \
                lambda: self._total_episodes_counter < self._n_episodes

            steps_progress_bar = tqdm(disable=True)
            episodes_progress_bar = tqdm(total=self._n_episodes,
                                         dynamic_ncols=True, disable=quiet,
                                         leave=False)

        return self._run_impl(move_condition, fit_condition, steps_progress_bar,
                              episodes_progress_bar, render, initial_states)

    def _run_impl(self, move_condition, fit_condition, steps_progress_bar,
                  episodes_progress_bar, render, initial_states):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0

        dataset = []
        last = None
        while move_condition():
            if last is None or np.sum(last) == len(last):
                self.reset(initial_states)

            sample = self._step(render)
            last_new = sample[-1]
            # if last is not None:
            #     sample = [[sample[i][j] for i in range(len(sample))] for j in np.where(last == 0)[0]]
            # else:
            #     sample = [[sample[i][j] for i in range(len(sample))] for j in range(len(sample[0]))]

            self.callback_step(sample)

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            if last is None or np.sum(last) == len(last):
                self._total_episodes_counter += 1
                self._current_episodes_counter += 1
                episodes_progress_bar.update(1)

            dataset.append(sample)
            if fit_condition():
                self.controller.fit(dataset)
                self._current_episodes_counter = 0
                self._current_steps_counter = 0

                for c in self.callbacks_fit:
                    c(dataset)

                dataset = []

            last = last_new

        self.controller.stop()
        self.mdp.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return dataset

    # Todo: add planner
    def _step(self, render):
        """
        Single step.

        Args:
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the
            agent, the reward obtained, the reached state, the absorbing flag
            of the reached state and the last step flag.

        """
        if self._actions is None:
            actions, values, log_probs = self.controller.draw_action(self._state)
            clipped_actions = np.clip(actions.cpu().numpy(), self.controller.policy.action_space.low, self.controller.policy.action_space.high)
            mask = np.ones(clipped_actions.shape[0])
        else:
            actions, clipped_actions, values, log_probs = self._actions, self._clipped_actions, self._values, self._log_probs
            mask = self._mask
        clipped_actions *= np.expand_dims(mask, axis=1).repeat(2, axis=1)
        union_actions = {'actions': clipped_actions, 'new_targets': []}

        next_state, reward, done, info = self.mdp.step(union_actions)

        self._mask = np.logical_not(done)
        self._episode_steps += 1

        if render:
            self.mdp.render()

        last = np.logical_or(self._episode_steps >= self.mdp.info.horizon, done)

        state = self._state

        next_actions, next_values, next_log_probs = self.controller.draw_action(next_state)
        next_clipped_actions = np.clip(next_actions.cpu().numpy(), self.controller.policy.action_space.low,
                                  self.controller.policy.action_space.high)

        self._state = next_state
        self._actions = next_actions
        self._clipped_actions = next_clipped_actions
        self._values = next_values
        self._log_probs = next_log_probs

        return state, actions, reward, log_probs, values, next_values, mask, done, last

    def reset(self, initial_states=None):
        """
        Reset the state of the agent.

        """
        if initial_states is None \
                or self._total_episodes_counter == self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        self._state = self._preprocess(self.mdp.reset(initial_state).copy())
        self.controller.next_action = None
        self._episode_steps = 0

        self._actions = None
        self._values = None
        self._log_probs = None
        self._clipped_actions = None
        self._mask = None

    def _preprocess(self, state):
        """
        Method to apply state preprocessors.

        Args:
            state (np.ndarray): the state to be preprocessed.

        Returns:
             The preprocessed state.

        """
        for p in self._preprocessors:
            state = p(state)

        return state
