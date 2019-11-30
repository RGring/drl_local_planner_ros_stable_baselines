'''
    @name:      ros_env.py
    @brief:     This (abstract) class is a simulation environment wrapper.
                It communicates with the BaseLocalPlanner in ROS and
                provides all relevant methods for the RL-library stable baselines.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''


# python relevant
import math
import random
import numpy as np
import gym
import time

# ros-relevant
import rospy

# messages
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from rl_msgs.msg import Waypoint
from geometry_msgs.msg import TwistStamped, Twist, Pose
from nav_msgs.msg import OccupancyGrid

# Helper classes
from rl_agent.env_utils.debug_ros_env import DebugRosEnv
from rl_agent.env_utils.reward_container import RewardContainer
from rl_agent.env_utils.task_generator import TaskGenerator
class RosEnvAbs(gym.Env):
    '''
    This (abstract) class is a simulation environment wrapper.
    It communicates with the BaseLocalPlanner in ROS and
    provides all relevant methods for the RL-library stable baselines.
    '''
    def __init__(self, ns, state_collector, execution_mode, task_mode, state_size, observation_space, stack_offset, action_size, action_space, debug, goal_radius, wp_radius, robot_radius, reward_fnc):
        super(RosEnvAbs, self).__init__()
        rospy.init_node("ros_env_%s"%ns, anonymous=True)

        #Setting random seed.
        seed = random.randint(0,1000)
        random.seed(seed)
        np.random.seed(seed)

        # Class variables
        self.STATE_SIZE = state_size
        self.observation_space = observation_space
        self.ACTION_SIZE = action_size
        self.action_space = action_space

        self.NS = ns                                    # namespace
        self.MODE = execution_mode                      # mode: TRAIN, EXEC, EXEC_RW, EVAL

        self.input_img_ = OccupancyGrid()               # occupancy grid containing state information
        self.twist_ = TwistStamped()                    # speed of robot
        self.__trigger = False                          # triggers agent to get state and compute action
        self.debug_ = debug                             # enable debugging
        self.done_ = True                              # Episode done?
        self.merged_scan_ = LaserScan()

        if self.MODE == "train" or self.MODE == "eval":
            self.__task_mode = task_mode                # "ped", "static", "toggle_ped_static", "ped_static"
            self.__toggle = True

            self.GOAL_RADIUS = goal_radius              # radius, when goal is reached
            self.WP_RADIUS = wp_radius                  # radius, when waypoint is reached
            self.ROBOT_RADIUS = robot_radius            # radius of the robot
            self.REWARD_FUNC = reward_fnc               # which reward function should be used.

            self.__is_env_closed = False
            self.__num_iterations = 0                   # counting number of iterations for each episode
            self.__transformed_goal = Pose()            # Goal in robot frame
            self.wp_ = Waypoint()                       # next waypoints on global plan
            self.static_scan_ = LaserScan()             # most recent static Laserscan
            self.ped_scan_ = LaserScan()                # most recent pedestrian Laserscan



        # Helper Classes
        self.__state_collector = state_collector
        if self.debug_:
            self.debugger_ = DebugRosEnv(self.NS, stack_offset)
        if self.MODE == "train" or self.MODE == "eval":
            if len(self.action_space.shape) == 0:
                self.__reward_cont = RewardContainer(self.NS, robot_radius, goal_radius, self.v_max_)
            else:
                self.__reward_cont = RewardContainer(self.NS, robot_radius, goal_radius, self.action_space.high[0])
            self.__task_generator = TaskGenerator(self.NS, self.__state_collector, self.ROBOT_RADIUS)

        # Subscriber
        self.__trigger_sub = rospy.Subscriber("%s/trigger_agent" % (self.NS), Bool, self.__trigger_callback)

        # Publisher
        self.__agent_action_pub = rospy.Publisher('%s/rl_agent/action' % (self.NS), Twist, queue_size=1)

        # Sleeping so that py-Publisher has time to setup!
        time.sleep(2)

    def step(self, action):
        """
        Action is forwarded to simulation. As reaction a new state is received.
        Depending on state observation, reward and done is computed.
        :return: obs, reward, done, info
        """
        # Publishing action
        # start = time.time()
        action = self.get_cmd_vel_(action)
        self.__agent_action_pub.publish(action)
        # print("publish cmd_vel: %f"%(time.time() - start))

        # waiting for robot-cycle to end
        begin = time.time()
        while not self.__trigger:
            if self.MODE == "train" or self.MODE == "eval":
                # Detecting if pipeline is broke --> resetting
                if (time.time() - begin) > 20.0:
                    rospy.logerr("%s, step(): Timeout while waiting for local planner." % (self.NS))
                    self.reset()
                    self.__agent_action_pub.publish(action)
                    begin = time.time()
            time.sleep(0.00001)
        self.__trigger = False
        # print("waiting for BaseLocalPlanner: %f"%(time.time() - begin))

        # start = time.time()
        self.__collect_state()
        # print("__collect_state: %f"%(time.time() - start))

        # start = time.time()
        obs = self.get_observation_()
        # print("get_observation_: %f"%(time.time() - start))
        info = {}
        if self.MODE == "train" or self.MODE == "train_demo" or self.MODE == "eval":
            # start = time.time()
            #info = [self.__num_iterations]
            action = np.array([action.linear.x, action.angular.z])
            reward = self.__compute_reward(action)
            self.done_, done_reason = self.__is_done(self.__num_iterations, self.static_scan_.ranges, self.ped_scan_.ranges, reward)
            info["done_reason"] = done_reason
            # print("reward, done, ...: %f" % (time.time() - start))
        else:
            reward = 0
            self.done_ = False

        return obs, reward, self.done_, info

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        self.__stop_robot()
        __is_env_closed = True

    def __set_task(self):
        """
        The task is set according to self.__task_mode and self.MODE.
        The different task_modes are the following:
        - ped: Spawning only pedestrians on path
        - static: Spawning only static obstacles on path
        - toggle_ped_static: Spawning alternating static obstacles and pedestrians
        - ped_static: Spawning static obstacles and pedestrians at the same time
        """
        if (self.MODE == "train" or self.MODE == "eval"):
            if self.__task_mode == "ped":
                self.__task_generator.set_random_ped_task()
            elif self.__task_mode == "ped_short":
                self.__task_generator.set_random_short_ped_task()
            elif self.__task_mode == "static":
                self.__task_generator.set_random_static_task()
            elif self.__task_mode == "toggle_ped_static":
                if self.__toggle:
                    self.__task_generator.set_random_static_task()
                else:
                    self.__task_generator.set_random_ped_task()
                self.__toggle = (not self.__toggle)
            elif self.__task_mode == "ped_static":
                self.__task_generator.set_random_static_ped_task()


    def reset(self):
        """
        Resetting simulation environment. That means, the robots position and goal
        are set randomly. Furthermore obstacles are spawned on that path.
        :return: initial observation of episode
        """
        if self.MODE != "train" and self.MODE != "train_demo" and self.MODE != "eval":
            return self.get_observation_()

        # resetting task
        self.__set_task()
        self.__agent_action_pub.publish(Twist())

        # waiting for planner to be ready for its first action
        begin = time.time()
        while not self.__trigger and not rospy.is_shutdown():
            if (time.time() - begin) > 20.0:
                rospy.logerr("%s, reset(): Timeout while waiting for local planner." % (self.NS))
                self.__set_task()
                begin = time.time()
            time.sleep(0.001)

        self.__trigger = False

        # reseting internal state values
        self.__num_iterations = 0

        # Initializing state variables
        self.__task_generator.take_sim_step()
        self.__collect_state()
        self.__reward_cont.reset(self.wp_.points)

        return self.get_observation_()

    def get_observation_(self):
        """
        Function returns state that will be fed to the rl-agent
        It includes laserscan data, waypoint information, ...
        :param
        :return: state
        """
        raise NotImplementedError("Not implemented!")

    def __collect_state(self):
        """
        State is collected.
        It can include the following information
        image containing laser data and waypoint
        goal in robot frame
        velocity of robot
        raw laser scans
        waypoints in robot frame.
        """
        [self.static_scan_, self.ped_scan_, self.merged_scan_, self.input_img_, self.wp_, self.twist_, self.__transformed_goal] = self.__state_collector.get_state()
        if(self.debug_):
            self.debugger_.show_wp(self.wp_)

    def __compute_reward(self, action):
        """
        Reward function gives feedback on taken action of the agent.
        :param
        :return: reward
        """
        if self.REWARD_FUNC == 1:
            reward = self.__reward_cont.rew_func_1(self.merged_scan_, self.wp_, self.__transformed_goal)
        elif self.REWARD_FUNC == 2.1:
            reward = self.__reward_cont.rew_func_2_1(self.static_scan_, self.ped_scan_, self.wp_, self.twist_,
                                                   self.__transformed_goal)
        elif self.REWARD_FUNC == 2.2:
            reward = self.__reward_cont.rew_func_2_2(self.static_scan_, self.ped_scan_, self.wp_, self.twist_, self.__transformed_goal)
        elif self.REWARD_FUNC == 19:
            reward = self.__reward_cont.rew_func_19(self.static_scan_, self.ped_scan_, self.wp_, self.twist_,
                                                    self.__transformed_goal)

        else:
            raise NotImplementedError

        if self.debug_:
            self.debugger_.show_reward(reward)
        return reward

    def __is_done(self, num_iterations, static_scan, ped_scan, reward):
        """
        Checks if end of episode is reached. It is reached,
        if maximum number of episodes is reached,
        if the goal is reached,
        if the robot collided with obstacle
        if the reward function returns a high negative value.
        :param current state
        :return: reward
        """
        # Is goal reached?
        dist_to_goal = self.mean_sqare_dist_(self.__transformed_goal.position.x,
                                             self.__transformed_goal.position.y)
        # Obstacle collision?
        min_obstacle_dist = np.amin(self.merged_scan_.ranges)

        # Ped in box?
        # is_ped_in_box = self.__reward_cont.is_ped_in_box(self.ped_scan_, 0.66, 0.86, 0.46)
        if self.MODE == "eval":
            max_iteration = 2000
        else:
            max_iteration = 650

        if dist_to_goal < self.GOAL_RADIUS:
            return [True, 1]
        if  min_obstacle_dist < self.ROBOT_RADIUS:
            return [True, 2]
        elif reward < -5:
            return [True, 5]
        elif num_iterations > max_iteration:
            return [True, 3]
        # elif self.MODE == "train" and self.mean_sqare_dist_(self.wp_.points[0].x, self.wp_.points[0].y) > 4:
        #     return [True, 4]

        self.__num_iterations += 1
        return [False, 0]


    def __trigger_callback(self, data):
        """
        If trigger is True, robot executed action successfully and provides a new state.
        The agent can now determine the next action.
        """

        self.__trigger = data.data
        return

    def mean_sqare_dist_(self, x, y):
        """
        Computing mean square distance of x and y
        :param x, y
        :return: sqrt(x^2 + y^2)
        """
        return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

    def get_cmd_vel_(self, action):
        """
        Decoding action (from rl_agent) to cmd_vel message
        :param action
        :return: cmd_vel message
        """
        raise NotImplementedError("Not implemented!")

    def get_action_list(self):
        """
        Getter for the action list. Is empty, when continuous action space
        :return: action list
        """
        raise NotImplementedError("Not implemented!")

    def get_goal_radius(self):
        """
        Getter for goal radius.
        :return: goal radius
        """
        return self.GOAL_RADIUS

    def get_wp_radius(self):
        """
        Getter for waypoint radius.
        :return: waypoint radius
        """
        return self.WP_RADIUS

    def get_state_size(self):
        """
         Getter for state size.
         :return: state size
         """
        return self.STATE_SIZE

    def get_action_size(self):
        """
        Getter for action size. Is inf if continuous
        :return: action size
        """
        return self.ACTION_SIZE
