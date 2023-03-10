import numpy as np
from igibson.termination_conditions.termination_condition_base import BaseTerminationCondition
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.utils.utils import l2_distance


class MaxCollision(BaseTerminationCondition):
    """
    MaxCollision used for navigation tasks
    Episode terminates if the robot has collided more than
    max_collisions_allowed times
    """

    def __init__(self, config):
        super(MaxCollision, self).__init__(config)
        self.max_collisions_allowed = self.config.get("max_collisions_allowed", 500)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if the robot has collided more than self.max_collisions_allowed times

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = env.collision_step > self.max_collisions_allowed
        success = np.zeros_like(done)
        return done, success


class Timeout(BaseTerminationCondition):
    """
    Timeout
    Episode terminates if max_step steps have passed
    """

    def __init__(self, config):
        super(Timeout, self).__init__(config)
        self.max_step = self.config.get("max_step", 500)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if max_step steps have passed

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = env.current_step >= self.max_step
        success = np.zeros_like(done)
        return done, success


class OutOfBound(BaseTerminationCondition):
    """
    OutOfBound used for navigation tasks in InteractiveIndoorScene
    Episode terminates if the robot goes outside the valid region
    """

    def __init__(self, config):
        super(OutOfBound, self).__init__(config)
        self.fall_off_thresh = self.config.get("fall_off_thresh", 0.03)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if the robot goes outside the valid region

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """

        done = np.zeros(env.num_robots)
        # fall off the cliff of valid region
        if isinstance(env.scene, InteractiveIndoorScene):
            for i in range(env.num_robots):
                if (not task.done[-1][i]) and env.robots[i].get_position()[2] < (env.scene.get_floor_height() - self.fall_off_thresh):
                    done[i] = True
        success = np.zeros_like(done)
        return done, success


class PointGoal(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(PointGoal, self).__init__(config)
        self.dist_tol = self.config.get("dist_tol", 0.5)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = np.zeros(env.num_robots)
        for i in range(env.num_robots):
            if not task.done[-1][i]:
                done[i] = l2_distance(env.robots[i].get_position()[:2], task.target_pos[i][:2]) < self.dist_tol
        success = done
        return done, success
