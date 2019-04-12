'''
    @name:      ros_env_cont_raw_data.py
    @brief:     This class is a simulation environment wrapper for
                the Raw Representation
                with continuous action space.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''


import rospy
# python relevant
import numpy as np
from gym import spaces

# custom classes
from rl_agent.env_wrapper.ros_env_raw_data import RosEnvRaw

# messages
from geometry_msgs.msg import Twist


# Parameters
GOAL_RADIUS = 0.4
WAYPOINT_RADIUS = 0.2

class RosEnvContRaw(RosEnvRaw):
    '''
    This class is a simulation environment wrapper for
    the X-Image Representation
    with continuous action space.
    '''
    def __init__(self, ns, state_collector, robot_radius = 0.46, reward_fnc=6, debug=False, execution_mode="train", task_mode="static"):
        state_size_t = rospy.get_param("%s/rl_agent/scan_size"% ns) + rospy.get_param("%s/rl_agent/num_of_wps"%ns)*2
        state_size = (1, state_size_t, 1)
        observation_space = spaces.Box(low=0, high=6, shape=state_size, dtype=np.float)
        action_space = spaces.Box(low=np.array([0.0, -0.5]), high=np.array([0.5, 0.5]), dtype=np.float)

        super(RosEnvContRaw, self).__init__(ns, state_collector, execution_mode, task_mode, state_size, observation_space, [], action_space, debug, GOAL_RADIUS, WAYPOINT_RADIUS, robot_radius, reward_fnc)

    def get_cmd_vel_(self, action):
        vel_msg = Twist()
        vel_msg.linear.x = action[0]
        vel_msg.angular.z = action[1]
        return vel_msg

    def get_action_list(self):
        action_list = []
        return action_list
