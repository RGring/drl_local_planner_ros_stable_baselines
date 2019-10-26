'''
    @name:      ros_env_img_vel.py
    @brief:     This (abstract) class is a simulation environment wrapper for
                the X-Image Speed Representation.
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

class RosEnvImgVel(RosEnvAbs):
    '''
    This (abstract) class is a simulation environment wrapper for
    the X-Image Speed Representation.
    '''
    def __init__(self, ns, state_collector, execution_mode, task_mode, state_size, observation_space, stack_offset, action_size, action_space, debug, goal_radius, wp_radius, robot_radius, reward_fnc):
        state_collector.set_state_mode(0)
        super(RosEnvImgVel, self).__init__(ns, state_collector, execution_mode, task_mode, state_size, observation_space, stack_offset, action_size, action_space, debug, goal_radius, wp_radius, robot_radius, reward_fnc)


    def get_observation_(self):
        """
        Function returns state that will be fed to the rl-agent
        It includes
        the laserscan and waypoint data in form of an image,
        the last speed of the robot, that is raw and stored in the last row in the state matrix.
        :return: state
        """
        obs = np.zeros(self.STATE_SIZE, dtype=np.float)
        recent_img = np.array(self.input_img_.data).reshape((self.STATE_SIZE[0] - 1, self.STATE_SIZE[1]))
        obs[0:-1,:,0] = recent_img
        obs[-1, 0:2, 0] = [self.twist_.twist.linear.x, self.twist_.twist.angular.z]
        if self.debug_:
            self.debugger_.show_input_occ_grid(self.input_img_)
            self.debugger_.show_image_stack(obs[:-1, :, :])
        return obs

