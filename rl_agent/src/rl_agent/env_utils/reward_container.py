'''
    @name:      ros_env.py
    @brief:     This class provides different reward functions and parametrizes them.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''


import numpy as np
import math
import rospy



class RewardContainer():
    """
    This class provides different reward functions and parametrizes them.
    """
    def __init__(self, ns, robot_radius, goal_radius, max_trans_vel):
        self.old_wps = []
        self.old_closest_wp = 0
        self.__update_rate = 1/rospy.get_param("%s/rl_agent/update_frequency"%ns)

        self.__robot_radius = robot_radius
        self.__goal_radius = goal_radius
        self.__still_time = 0.0
        self.__max_trans_vel = max_trans_vel


    def reset(self, wps):
        """
        If episode is reset, old waypoints need to be resetted here.
        """
        num_wp = len(wps)
        self.old_wps = np.zeros(num_wp)
        for i in range(num_wp):
            self.old_wps[i] = self.mean_sqare_dist_(wps[i].x, wps[i].y)
        self.old_closest_wp = 0

    def rew_func_1(self, scan, wps, transformed_goal):
        '''
        Reward function designed for static training setup
        :param scan: laser scan of environment
        :param wps: next waypoints on path
        :param transformed_goal: final goal in robot frame
        :return: reward value
        '''
        min_scan_dist = np.amin(scan.ranges)

        if (min_scan_dist < (self.__robot_radius + 0.5) ):
            wp_approached_rew = self.__get_wp_approached(wps, 3.5, 2.5, 1.0)
            wp_approached_rew = 0
        else:
            wp_approached_rew = self.__get_wp_approached(wps, 3.5, 2.5, 1.0)

        obstacle_punish = self.__get_obstacle_punish(scan.ranges , 15, self.__robot_radius)
        goal_reached_rew = self.__get_goal_reached_rew(transformed_goal, 10)
        rew = (wp_approached_rew + obstacle_punish + goal_reached_rew)
        rew = self.__check_reward(rew, obstacle_punish, goal_reached_rew, 2.5)
        return rew

    def rew_func_19(self, static_scan, ped_scan_msg, wps, twist, transformed_goal):
        '''
                Reward function designed for dynamic training setup
                :param static_scan: laser scan containing information about static obstacles.
                :param ped_scan_msg: laser scan containing information about dynamic obstacles.
                :param wps: next waypoints on path
                :param twist: velocity of robot
                :param transformed_goal: final goal in robot frame
                :return: reward value
                '''
        standing_still_punish = 0
        if (abs(twist.twist.linear.x) < 0.001 and abs(twist.twist.angular.z) < 0.001):
            standing_still_punish = -0.001
            self.__still_time += 0.1
        else:
            self.__still_time = 0.0

        if (abs(twist.twist.linear.x) < 0.001 and abs(twist.twist.angular.z) > 0.001):
            standing_still_punish = -0.01

        wp_approached_rew = self.__get_wp_approached(wps, 5.5, 4.5, 0.0)

        # Did the agent bump into an obstacle?
        obstacle_punish_static = self.__get_obstacle_punish(static_scan.ranges, 7, self.__robot_radius)
        obstacle_punish_ped = 0
        if (self.__still_time < 0.8):
            obstacle_punish_ped = self.__get_obstacle_punish(ped_scan_msg.ranges, 7, 0.85)
        obstacle_punish = min(obstacle_punish_ped, obstacle_punish_static)

        # Did the agent reached the goal?
        goal_reached_rew = self.__get_goal_reached_rew(transformed_goal, 10)

        rew = (wp_approached_rew + obstacle_punish + goal_reached_rew + standing_still_punish)
        if (rew < -2.5):
            test = "debug"
        rew = self.__check_reward(rew, obstacle_punish, goal_reached_rew, 2.5)
        return rew

    def rew_func_2_1(self, static_scan, ped_scan_msg, wps, twist, transformed_goal):
        '''
        Reward function designed for dynamic training setup
        :param static_scan: laser scan containing information about static obstacles.
        :param ped_scan_msg: laser scan containing information about dynamic obstacles.
        :param wps: next waypoints on path
        :param twist: velocity of robot
        :param transformed_goal: final goal in robot frame
        :return: reward value
        '''
        standing_still_punish = 0
        if (abs(twist.twist.linear.x) < 0.001 and abs(twist.twist.angular.z) < 0.001):
            standing_still_punish = -0.001
            self.__still_time += 0.1
        else:
            self.__still_time = 0.0

        if (abs(twist.twist.linear.x) < 0.001 and abs(twist.twist.angular.z) > 0.001):
            standing_still_punish = -0.01
        min_scan_dist = np.amin(static_scan.ranges)
        if (min_scan_dist < (self.__robot_radius + 0.2)):
            wp_approached_rew = self.__get_wp_approached(wps, 5.5, 4.5, 0.8)
            wp_approached_rew = 0
        else:
            wp_approached_rew = self.__get_wp_approached(wps, 5.5, 4.5, 0.8)

        # Did the agent bump into an obstacle?
        obstacle_punish_static = self.__get_obstacle_punish(static_scan.ranges, 15, self.__robot_radius)
        obstacle_punish_ped = 0
        if (self.__still_time < 0.8):
            obstacle_punish_ped = self.__get_obstacle_punish(ped_scan_msg.ranges, 7, 0.85)
        obstacle_punish = min(obstacle_punish_ped, obstacle_punish_static)

        # Did the agent reached the goal?
        goal_reached_rew = self.__get_goal_reached_rew(transformed_goal, 10)

        rew = (wp_approached_rew + obstacle_punish + goal_reached_rew + standing_still_punish)
        if (rew < -2.5):
            test = "debug"
        rew = self.__check_reward(rew, obstacle_punish, goal_reached_rew, 2.5)
        return rew


    def rew_func_2_2(self, static_scan, ped_scan_msg, wps, twist, transformed_goal):
        '''
        Reward function designed for dynamic training setup
        :param static_scan: laser scan containing information about static obstacles.
        :param ped_scan_msg: laser scan containing information about dynamic obstacles.
        :param wps: next waypoints on path
        :param twist: velocity of robot
        :param transformed_goal: final goal in robot frame
        :return: reward value
        '''
        standing_still_punish = 0
        if (abs(twist.twist.linear.x) < 0.1):
            self.__still_time += 0.1
        else:
            self.__still_time = 0.0
        min_scan_dist = np.amin(static_scan.ranges)

        if (min_scan_dist < (self.__robot_radius + 0.2)):
            wp_approached_rew = self.__get_wp_approached(wps, 5.5, 4.5, 0.8)
            wp_approached_rew = 0
        else:
            wp_approached_rew = self.__get_wp_approached(wps, 5.5, 4.5, 0.8)

        # Did the agent bump into an obstacle?
        obstacle_punish_static = self.__get_obstacle_punish(static_scan.ranges, 15, self.__robot_radius)
        obstacle_punish_ped = 0
        if (self.__still_time < 0.8):
            obstacle_punish_ped = self.__get_obstacle_punish(ped_scan_msg.ranges, 7, 0.85)
        obstacle_punish = min(obstacle_punish_ped, obstacle_punish_static)

        # Did the agent reached the goal?
        goal_reached_rew = self.__get_goal_reached_rew(transformed_goal, 10)

        rew = (wp_approached_rew + obstacle_punish + goal_reached_rew + standing_still_punish)
        if (rew < -2.5):
            test = "debug"
        rew = self.__check_reward(rew, obstacle_punish, goal_reached_rew, 2.5)
        return rew


    def __ped_in_box_punish(self, ped_scan_msg, box_width, box_height_pos, box_height_neg, k):
        """
        Returns a negative reward k if pedestrians are in the defined box
        [box_width x (box_height_pos + box_height_neg)}
        :param ped_scan_msg laserscan with pedestrian information only
        :param box_width width if box
        :param box_height_pos height of box in forward direction of the robot
        :param box_height_neg height of box in backward direction of the robot
        :param k reward constant
        :return: obs, reward, done, info
        """
        if self.is_ped_in_box(ped_scan_msg, box_width, box_height_pos, box_height_neg):
            ped_punish = -k
        else:
            ped_punish = 0

        return ped_punish

    def is_ped_in_box(self, ped_scan_msg, box_width, box_height_pos, box_height_neg):
        '''
        Checks if pedestrian is in defined box [box_width x (box_height_pos + box_height_neg)]
        :param ped_scan_msg laserscan with pedestrian information only
        :param box_width width if box
        :param box_height_pos height of box in forward direction of the robot
        :param box_height_neg height of box in backward direction of the robot
        :return: True if pedestrian inside the defined box, else False.
        '''
        is_ped_in_box = False
        box_width = box_width * 2
        box_height_pos = box_height_pos
        box_height_neg = box_height_neg
        angle_min = ped_scan_msg.angle_min
        angle_increment = ped_scan_msg.angle_increment
        ped_punish = 0.0
        for i in range(len(ped_scan_msg.ranges)):
            angle = angle_min + i * angle_increment
            length = ped_scan_msg.ranges[i]
            x = math.cos(angle) * length
            y = math.sin(angle) * length
            if(x > -box_height_neg and x < box_height_pos and y > -box_width/2 and y < box_width/2):
                is_ped_in_box = True
        return is_ped_in_box

    def __check_reward(self, rew, obstacle_punish, goal_reached_rew, thresh):
        """
        Checks if reward makes sense.
        :param rew final reward
        :param obstacle_punish reward for obstacle collision
        :param goal_reached_rew reward for reaching goal
        :param thresh [-thresh, +thresh[ the final reward should be inbetween, if no obstacle_punish==0 and goal_reached_rew==0
        :return: returns reward value if it makes sense, else 0
        """
        if ((rew > thresh and goal_reached_rew == 0.0) or (rew < -thresh and obstacle_punish == 0.0)):
            rospy.loginfo("Wrong reward: %f" % rew)
            return 0
        else:
            return rew

    def __get_turn_punish(self, w,  fac, thresh):
        """
        Returns negative reward if the robot turns.
        :param w roatational speed of the robot
        :param fac weight of reward punish for turning
        :param thresh rotational speed > thresh will be punished
        :return: returns reward for turning
        """
        if abs(w) > thresh:
            return abs(w) * -fac
        else:
            return 0.0

    def __get_obstacle_punish(self, scan, k, radius):
        """
        Returns negative reward if the robot collides with obstacles.
        :param scan containing obstacles that should be considered
        :param k reward constant
        :return: returns reward colliding with obstacles
        """
        min_scan_dist = np.amin(scan)

        if min_scan_dist < radius:
            return -k
        else:
            return 0.0

    def __get_obstacle_punish_section(self, scan, k, perc):
        """
        Returns negative reward if the robot collides with obstacles in section around the robot.
        :param scan containing obstacles that should be considered
        :param k reward constant
        :param perc percentage of the area around the robot, that should be considered.
        Example: 0.4 will consider the front view of 144 degree.
        :return: returns reward colliding with obstacles in defined section
        """
        min_scan_dist_index = np.argmin(scan)
        scan_length = len(scan)
        lower_bound = int(scan_length/2 - perc*scan_length/2)
        upper_bound = int(scan_length/2 + perc*scan_length/2)
        if min_scan_dist_index > lower_bound and min_scan_dist_index < upper_bound and scan[min_scan_dist_index] < self.__robot_radius:
            return -k
        else:
            return 0.0

    def __get_goal_reached_rew(self, transformed_goal, k):
        """
        Returns positive reward if the robot reaches the goal.
        :param transformed_goal goal position in robot frame
        :param k reward constant
        :return: returns reward colliding with obstacles
        """
        dist_to_goal = self.mean_sqare_dist_(transformed_goal.position.x, transformed_goal.position.y)
        if dist_to_goal < self.__goal_radius:
            return k
        else:
            return 0.0


    def __get_wp_approached(self, wps, punish_fac, rew_fac, k):
        """
        Returns positive reward if the robot approaches the next waypoint, else negative reward.
        :param wps next waypoints
        :param punish_fac weight for punishing disapproaching the closest waypoint
        :param rew_fac weight for rewarding approaching the closest waypoint
        :param k reward constant for reaching a waypoint
        :return: returns reward for approaching waypoints.
        """
        num_wp = len(wps.points)
        dist_to_waypoint = np.zeros(num_wp)

        for i in range(num_wp):
            dist_to_waypoint[i] = self.mean_sqare_dist_(wps.points[i].x, wps.points[i].y)

        if wps.is_new.data:
            wp_approached =  k
        else:
            diff = (self.old_wps[0] - dist_to_waypoint[0])
            if (abs(diff) > self.__max_trans_vel*self.__update_rate*4):
                diff = 0
            if (diff < 0):
                wp_approached = punish_fac * diff
            else:
                wp_approached = rew_fac * diff

        self.old_wps = dist_to_waypoint

        return wp_approached

    def mean_sqare_dist_(self, x, y):
        """
        Computing mean square distance of x and y
        :param x, y
        :return: sqrt(x^2 + y^2)
        """
        return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

