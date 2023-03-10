import numpy as np
from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance


class PotentialReward(BaseRewardFunction):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)
    """

    def __init__(self, config):
        super(PotentialReward, self).__init__(config)
        self.potential_reward_weight = self.config.get("potential_reward_weight", 1.0)

    def reset(self, task, env):
        """
        Compute the initial potential after episode reset

        :param task: task instance
        :param env: environment instance
        """
        self.potential = task.get_potential(env)

    def reset_for_robot(self, task, env, robot_id):
        new_potential = task.get_potential(env)
        self.potential[robot_id] = new_potential[robot_id]

    def get_reward(self, task, env):
        """
        Reward is proportional to the potential difference between
        the current and previous timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        new_potential = task.get_potential(env)
        reward = self.potential - new_potential
        reward = self.potential_reward_weight * reward
        self.potential = new_potential
        return reward


class CollisionReward(BaseRewardFunction):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative.
    """

    def __init__(self, config):
        super(CollisionReward, self).__init__(config)
        self.collision_reward_weight = self.config.get("collision_reward_weight", -0.1)

    def reset_for_robot(self, task, env, robot_id):
        return

    def get_reward(self, task, env):
        """
        Reward is self.collision_reward_weight if there is collision
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        has_collision = np.array([(float(len(env.collision_links[i]) > 0) if not task.done[-1][i] else 0) for i in range(env.num_robots)])
        return has_collision * self.collision_reward_weight


class TimeReward(BaseRewardFunction):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative.
    """

    def __init__(self, config):
        super(TimeReward, self).__init__(config)
        self.time_reward_weight = self.config.get("time_reward_weight", -0.1)

    def reset_for_robot(self, task, env, robot_id):
        return

    def get_reward(self, task, env):
        """
        Reward is self.collision_reward_weight if there is collision
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        return np.ones(env.num_robots) * self.time_reward_weight


class PointGoalReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(PointGoalReward, self).__init__(config)
        self.success_reward = self.config.get("success_reward", 10.0)
        self.dist_tol = self.config.get("dist_tol", 0.5)

    def reset_for_robot(self, task, env, robot_id):
        return

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """

        success = [(l2_distance(env.robots[i].get_position()[:2], task.target_pos[i][:2]) < self.dist_tol and not task.done[-1][i]) for i in
                   range(env.num_robots)]
        reward = np.zeros(env.num_robots) + np.array(success) * self.success_reward

        return reward
