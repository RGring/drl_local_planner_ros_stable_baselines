'''
    @name:      ros_env_img.py
    @brief:     This (abstract) class is a simulation environment wrapper for
                the X-Image Representation.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''


# python relevant
import numpy as np

# custom classes
from rl_agent.env_wrapper.ros_env import RosEnvAbs

# ros-relevant
import rospy

class RosEnvImg(RosEnvAbs):
    '''
    This (abstract) class is a simulation environment wrapper for
    the X-Image Representation.
    '''
    def __init__(self, ns, state_collector, execution_mode, task_mode, state_size, observation_space, stack_offset, action_size, action_space, debug, goal_radius, wp_radius, robot_radius, reward_fnc):
        state_collector.set_state_mode(0)
        super(RosEnvImg, self).__init__(ns, state_collector, execution_mode, task_mode, state_size, observation_space, stack_offset, action_size, action_space, debug, goal_radius, wp_radius, robot_radius, reward_fnc)


    def get_observation_(self):
        """
        Function returns state that will be fed to the rl-agent
        It includes
        the laserscan and the waypoint information stored in an image.
        :return: state
        """
        obs = np.zeros(self.STATE_SIZE, dtype=np.float)
        obs[:,:,0] = np.array(self.input_img_.data).reshape((self.STATE_SIZE[0:2]))

        if self.debug_:
            self.debugger_.show_input_occ_grid(self.input_img_)
            self.debugger_.show_input_image(obs[:,:,0])
        return obs

