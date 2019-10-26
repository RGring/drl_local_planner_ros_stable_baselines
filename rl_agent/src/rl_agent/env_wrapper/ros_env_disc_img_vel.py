'''
    @name:      ros_env_cont_img_vel.py
    @brief:     This class is a simulation environment wrapper for
                the X-Image Speed Representation
                with discrete action space.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''

# ros-relevant
import rospy

# python relevant
import numpy as np
from gym import spaces

#custom classes
from rl_agent.env_wrapper.ros_env_img_vel import RosEnvImgVel

# Messages
from geometry_msgs.msg import Twist


# Parameters
GOAL_RADIUS = 0.4
WAYPOINT_RADIUS = 0.2

class RosEnvDiscImgVel(RosEnvImgVel):
    '''
    This class is a simulation environment wrapper for
    the X-Image Speed Representation
    with discrete action space.
    '''
    def __init__(self, ns, state_collector, stack_offset, stack_size, robot_radius = 0.46, reward_fnc=6, debug=False, execution_mode="train", task_mode="static"):
        img_width = rospy.get_param("%s/rl_agent/img_width_pos" % ns) + rospy.get_param(
            "%s/rl_agent/img_width_neg" % ns)
        img_height = rospy.get_param("%s/rl_agent/img_height" % ns)
        state_size = (img_height + 1, img_width, 1)
        observation_space = spaces.Box(low=0, high=100, shape=state_size, dtype=np.float)

        self.v_max_ = 0.8 # ?1.5?
        self.w_max_ = 1.2
        self.__possible_actions = {
            0: [0.0, -self.w_max_],
            1: [self.v_max_, 0.0],
            2: [0.0, self.w_max_],
            3: [self.v_max_, self.w_max_/2],
            4: [self.v_max_, -self.w_max_/2],
            5: [0.0, 0.0]
            # 6: [0.09, 0.0]
        }
        action_size = len(self.__possible_actions)
        action_space = spaces.Discrete(action_size)
        super(RosEnvDiscImgVel, self).__init__(ns, state_collector, execution_mode, task_mode, state_size, observation_space, stack_offset, action_size, action_space, debug, GOAL_RADIUS, WAYPOINT_RADIUS, robot_radius, reward_fnc)

    def get_cmd_vel_(self, action):
        encoded_action = self.__encode_action(action)
        vel_msg = Twist()
        vel_msg.linear.x = encoded_action[0]
        vel_msg.angular.z = encoded_action[1]
        return vel_msg

    def __encode_action(self, action):
        return self.__possible_actions.get(action, 5)

    def get_action_list(self):
        action_list = []
        for i in range(self.ACTION_SIZE):
            action_list.append(self.__encode_action(i))
        return action_list
