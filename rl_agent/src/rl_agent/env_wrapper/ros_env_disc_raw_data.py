'''
    @name:      ros_env_cont_raw_data.py
    @brief:     This class is a simulation environment wrapper for
                the Raw Representation
                with discrete action space.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''

# python relevant
import numpy as np
from gym import spaces
import rospy
# custom classes
from rl_agent.env_wrapper.ros_env_raw_data import RosEnvRaw

# messages
from geometry_msgs.msg import Twist


# Parameters
GOAL_RADIUS = 0.4
WAYPOINT_RADIUS = 0.2



class RosEnvDiscRaw(RosEnvRaw):
    '''
    This class is a simulation environment wrapper for
    the Raw Representation
    with discrete action space.
    '''
    def __init__(self, ns, state_collector, robot_radius = 0.46, reward_fnc=6, debug=False, execution_mode="train", task_mode="static"):
        state_size_t = rospy.get_param("%s/rl_agent/scan_size"% ns) + rospy.get_param("%s/rl_agent/num_of_wps"%ns)*2
        state_size = (1, state_size_t, 1)
        observation_space = spaces.Box(low=0, high=6, shape=state_size, dtype=np.float)

        self.action = np.array([0.0, 0.0])
        v_max = 0.5
        w_max = 0.5
        self.__possible_actions = {
            0: [0.0, -w_max],
            1: [v_max, 0.0],
            2: [0.0, w_max],
            3: [v_max, w_max / 2],
            4: [v_max, -w_max / 2],
            5: [0.0, 0.0],
        }
        action_size = len(self.__possible_actions)

        action_space = spaces.Discrete(action_size)
        super(RosEnvDiscRaw, self).__init__(ns, state_collector, execution_mode, task_mode, state_size, observation_space, action_size, action_space, debug, GOAL_RADIUS, WAYPOINT_RADIUS, robot_radius, reward_fnc)
        self.action = np.array([0.0, 0.0])

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
