'''
    @name:      ros_env_cont_img_vel.py
    @brief:     This class is a simulation environment wrapper for
                the X-Image Speed Representation
                with continuous action space.
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


class RosEnvContImgVel(RosEnvImgVel):
    '''
    This class is a simulation environment wrapper for
    the X-Image Speed Representation
    with continuous action space.
    '''
    def __init__(self, ns, state_collector, stack_offset, stack_size, robot_radius = 0.46, reward_fnc=6,debug=False, execution_mode="train", task_mode="static"):
        img_width = rospy.get_param("%s/rl_agent/img_width_pos" % ns) + rospy.get_param(
            "%s/rl_agent/img_width_neg" % ns)
        img_height = rospy.get_param("%s/rl_agent/img_height" % ns)
        state_size = (img_height + 1, img_width , 1)
        observation_space = spaces.Box(low=0, high=100, shape=state_size, dtype=np.float)
        action_space = spaces.Box(low=np.array([0.0, -0.7]), high=np.array([0.5, 0.7]), dtype=np.float)
        super(RosEnvContImgVel, self).__init__(ns, state_collector, execution_mode, task_mode, state_size, observation_space, stack_offset, [], action_space, debug, GOAL_RADIUS, WAYPOINT_RADIUS, robot_radius, reward_fnc)

    def get_cmd_vel_(self, action):
        vel_msg = Twist()
        vel_msg.linear.x = action[0]
        vel_msg.angular.z = action[1]
        return vel_msg

    def get_action_list(self):
        action_list = []
        return action_list
