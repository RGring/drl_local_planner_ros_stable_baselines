'''
    @name:      ros_env_raw_data.py
    @brief:     This (abstract) class is a simulation environment wrapper for
                the Raw Representation.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''

# python relevant
import numpy as np
import rospy

# custom classes
from rl_agent.env_wrapper.ros_env import RosEnvAbs
from sensor_msgs.msg import LaserScan

class RosEnvRaw(RosEnvAbs):
    '''
    This (abstract) class is a simulation environment wrapper for
    the Raw Representation.
    '''
    def __init__(self, ns, state_collector, execution_mode, task_mode, state_size, observation_space, stack_offset, action_size, action_space, debug, goal_radius, wp_radius, robot_radius, reward_fnc):
        state_collector.set_state_mode(2)

        super(RosEnvRaw, self).__init__(ns, state_collector, execution_mode, task_mode, state_size, observation_space, stack_offset, action_size, action_space, debug, goal_radius, wp_radius, robot_radius, reward_fnc)
        self.__scan_size = rospy.get_param("%s/rl_agent/scan_size"%ns)

    def get_observation_(self):
        """
        Function returns state that will be fed to the rl-agent
        It includes
        the raw laser scan data,
        the raw waypoint data.
        :return: state
        """
        waypoint = self.wp_
        num_of_wps = len(waypoint.points)
        state = np.ones(self.STATE_SIZE, dtype=np.float)

        # add laserscan
        state[0, 0:self.__scan_size, 0] = self.merged_scan_.ranges

        # add goal position
        wp_index = self.STATE_SIZE[1] - num_of_wps * 2
        for i in range(num_of_wps):
            state[0, (wp_index + i*2):(wp_index + i*2 + 2),0] = [waypoint.points[i].x, waypoint.points[i].y]

        # Discretize to a resolution of 5cm.
        state = np.round(np.divide(state, 0.05))*0.05
        if self.debug_:
            debug_scan = LaserScan()
            debug_scan.header = self.merged_scan_.header
            debug_scan.angle_min = self.merged_scan_.angle_min
            debug_scan.angle_max = self.merged_scan_.angle_max
            debug_scan.angle_increment = self.merged_scan_.angle_increment
            debug_scan.range_max = 7.0
            debug_scan.ranges = state[0, 0:self.__scan_size, 0]
            self.debugger_.show_input_scan(debug_scan)
        return state
