import numpy as np
import pybullet as p
import logging

from igibson.objects.visual_marker import VisualMarker
from reward_functions import CollisionReward, PointGoalReward, PotentialReward, TimeReward
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.tasks.task_base import BaseTask
from termination_conditions import MaxCollision, OutOfBound, PointGoal, Timeout
from igibson.utils.utils import cartesian_to_polar, l2_distance, rotate_vector_3d


class MultiPointNavTask(BaseTask):
    """
    Point Nav Fixed Task
    The goal is to navigate to a fixed goal position
    """

    def __init__(self, env):
        super(MultiPointNavTask, self).__init__(env)
        self.reward_type = self.config.get("reward_type", "l2")
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            PointGoal(self.config),
        ]
        self.reward_functions = [
            PotentialReward(self.config),
            CollisionReward(self.config),
            PointGoalReward(self.config),
            TimeReward(self.config)
        ]

        self.num_robots = self.config['num_robots']
        self.max_trials = 500

        self.initial_pos = [None for _ in range(self.num_robots)]
        self.initial_orn = [None for _ in range(self.num_robots)]
        self.target_pos = [None for _ in range(self.num_robots)]
        self.goal_format = self.config.get("goal_format", "polar")
        self.dist_tol = self.termination_conditions[-1].dist_tol

        self.task_random = self.config.get("task_random", False)

        self.visual_object_at_initial_target_pos = self.config.get("visual_object_at_initial_target_pos", True)
        self.target_visual_object_visible_to_agent = self.config.get("target_visual_object_visible_to_agent", False)
        self.remove_finished_robots = self.config.get("remove_finished_robots", True)
        self.floor_num = self.config.get("floor_num", 0)
        self.target_dist_min = self.config.get("target_dist_min", 1.0)
        self.target_dist_max = self.config.get("target_dist_max", 10.0)

        self.load_visualization(env)

    def load_visualization(self, env):
        """
        Load visualization, such as initial and target position, shortest path, etc

        :param env: environment instance
        """
        # if env.mode != "gui":
        #     return

        cyl_length = 0.2
        self.initial_pos_vis_objs = [VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[1, 0, 0, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        ) for _ in range(self.num_robots)]
        self.target_pos_vis_objs = [VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 1, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        ) for _ in range(self.num_robots)]

        if self.target_visual_object_visible_to_agent:
            for i in range(self.num_robots):
                env.simulator.import_object(self.initial_pos_vis_objs[i])
                env.simulator.import_object(self.target_pos_vis_objs[i])
        else:
            for i in range(self.num_robots):
                self.initial_pos_vis_objs[i].load()
                self.target_pos_vis_objs[i].load()

    def get_geodesic_potential(self, env):
        """
        Get potential based on geodesic distance

        :param env: environment instance
        :return: geodesic distance to the target position
        """
        _, geodesic_dist = self.get_shortest_path(env)
        return geodesic_dist

    def get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        :param env: environment instance
        :return: L2 distance to the target position
        """
        return np.array(
            [(l2_distance(env.robots[i].get_position()[:2], self.target_pos[:2]) if not self.done[-1][i] else 0) for i in
             range(self.num_robots)])

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        :param env: environment instance
        :return: task potential
        """
        if self.reward_type == "l2":
            return self.get_l2_potential(env)
        elif self.reward_type == "geodesic":
            return self.get_geodesic_potential(env)

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset scene objects or floor plane

        :param env: environment instance
        """
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)

    def reset_position(self, initial_pos=None, target_pos=None, initial_orn=None):
        if initial_pos is not None:
            self.initial_pos = initial_pos
        if target_pos is not None:
            self.target_pos = target_pos
        if initial_orn is None:
            self.initial_orn = [np.array([0, 0, np.random.uniform(0, np.pi * 2)]) for _ in range(self.num_robots)]

    def reset_agent(self, env):
        """
        Task-specific agent reset: land the robot to initial pose, compute initial potential

        :param env: environment instance
        """
        self.done = [np.zeros(self.num_robots)]
        for i in range(self.num_robots):
            if not self.task_random:
                self.reset_target(env, i, self.target_pos[i], initializing=True)
            else:
                self.reset_target(env, i, initializing=True)

        for i in range(self.num_robots):
            if not self.task_random:
                reset_success = self.place_robot(env, i, self.initial_pos[i], self.initial_orn[i])
            else:
                for _ in range(self.max_trials):
                    _, pos = env.scene.get_random_point(floor=self.floor_num)
                    orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                    reset_success = env.test_valid_position(env.robots[i], pos, orn)
                    if not reset_success:
                        continue

                    dist = 0.0
                    if env.scene.build_graph:
                        _, dist = env.scene.get_shortest_path(
                            self.floor_num, pos[:2], self.target_pos[i][:2], entire_path=False
                        )
                    else:
                        dist = l2_distance(pos, self.target_pos[i])
                    if self.target_dist_min < dist < self.target_dist_max:
                        break

                self.initial_pos[i] = pos
                self.initial_orn[i] = orn

            if not reset_success:
                logging.warning("WARNING: Failed to reset robot without collision")

        self.initial_pos = np.array(self.initial_pos)
        self.target_pos = np.array(self.target_pos)
        self.path_length = np.zeros(self.num_robots)
        self.robot_pos = self.initial_pos[:, :2]
        self.geodesic_dist = self.get_geodesic_potential(env)
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done = np.zeros(env.num_robots)
        success = np.zeros(env.num_robots)
        for condition in self.termination_conditions:
            d, s = condition.get_termination(self, env)
            done = np.logical_or(d, done)
            success = np.logical_or(s, success)

        done = np.logical_or(self.done[-1], done)
        self.done.append(done)

        # if self.remove_finished_robots:
        #     self.remove_redundant_robots(env, self.done)

        info["done"] = done
        info["success"] = success
        return done, info

    # def remove_redundant_robots(self, env, done):
    #     if len(done) == 1:
    #         return
    #     for i in range(self.num_robots):
    #         if done[-1][i] and done[-2][i]:
    #             env.robots[i].set_position([100.0 * (i + 1), 100.0 * (i + 1), 100.0 * (i + 1)])

    def global_to_local(self, robot, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(np.array(pos) - np.array(robot.get_position()), *robot.get_rpy())

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        task_obs = [self.global_to_local(env.robots[i], self.target_pos[i])[:2] for i in range(self.num_robots)]
        if self.goal_format == "polar":
            task_obs = [np.array(cartesian_to_polar(task_obs[i][0], task_obs[i][1])) for i in range(self.num_robots)]

        task_obs = np.array(task_obs)
        velocity = np.array([[rotate_vector_3d(robot.get_linear_velocity(), *robot.get_rpy())[0],
                              rotate_vector_3d(robot.get_angular_velocity(), *robot.get_rpy())[2]] for robot in
                             env.robots])
        task_obs = np.concatenate([task_obs, velocity], axis=1)

        return task_obs

    def get_shortest_path(self, env, entire_path=False):
        """
        Get the shortest path and geodesic distance from the robot or the initial position to the target position

        :param env: environment instance
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        source = [robot.get_position()[:2] for robot in env.robots]
        target = self.target_pos[:, :2]

        shortest_paths = []
        geodesic_distances = []

        for i in range(self.num_robots):
            if not self.done[-1][i]:
                shortest_path, geodesic_distance = env.scene.get_shortest_path(self.floor_num, source[i], target[i],
                                                                               entire_path=entire_path)
            else:
                shortest_path, geodesic_distance = [], 0
            shortest_paths.append(shortest_path)
            geodesic_distances.append(geodesic_distance)

        return shortest_paths, np.array(geodesic_distances)

    def step_visualization(self, env):
        """
        Step visualization

        :param env: environment instance
        """
        # if env.mode != "gui":
        #     return

        for i in range(self.num_robots):
            if not self.done[-1][i]:
                self.initial_pos_vis_objs[i].set_position(self.initial_pos[i])
                self.target_pos_vis_objs[i].set_position(self.target_pos[i])

    def get_reward(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate reward functions

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return reward: total reward of the current timestep
        :return info: additional info
        """
        reward = np.zeros(env.num_robots)
        for reward_function in self.reward_functions:
            reward += reward_function.get_reward(self, env)

        return reward, info

    def reset_target(self, env, robot_index, target=None, initializing=False):
        if target is not None:
            self.target_pos[robot_index] = target
        else:
            for _ in range(self.max_trials):
                _, target = env.scene.get_random_point(floor=self.floor_num)
                if not env.test_valid_position(env.robots[robot_index], target):
                    continue

            self.target_pos[robot_index] = target

        if not initializing:
            self.done[-1][robot_index] = 0
            self.robot_pos = self.initial_pos[:, :2]
            self.geodesic_dist = self.get_geodesic_potential(env)
            for reward_function in self.reward_functions:
                reward_function.reset_for_robot(self, env, robot_index)

    def reset_robot_position(self, env, robot_index):
        last_target = self.target_pos[robot_index]
        heading = np.arctan2(last_target[1] - self.initial_pos[robot_index][1],
                             last_target[0] - self.initial_pos[robot_index][0])
        self.initial_orn[robot_index] = [0, 0, heading]

        reset_success = self.place_robot(env, robot_index, last_target, self.initial_orn[robot_index])

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        env.robots[robot_index].robot_specific_reset()
        env.robots[robot_index].keep_still()

    def place_robot(self, env, robot_index, target, orn):
        if not env.check_robots_collision(robot_index, target) and env.test_valid_position(env.robots[robot_index], target, orn):
            self.initial_pos[robot_index] = target
            reset_success = True
        else:
            for _ in range(self.max_trials):
                _, pos = env.get_random_point_near(floor=self.floor_num, base_point=target)
                if pos is None:
                    env.test_valid_position(env.robots[robot_index], target, orn)
                    return False
                reset_success = env.test_valid_position(env.robots[robot_index], pos, orn)
                if reset_success:
                    self.initial_pos[robot_index] = pos
                    break

        return reset_success
    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        self.step_visualization(env)
        new_robot_pos = [env.robots[i].get_position()[:2] for i in range(env.num_robots)]
        self.path_length += l2_distance(self.robot_pos, new_robot_pos)
        self.robot_pos = new_robot_pos
