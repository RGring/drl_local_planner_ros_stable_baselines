'''
    @name:      ros_env_raw_scan_prep_wo.py
    @brief:     This class is a simulation environment wrapper for
                the Polar Representation.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''

# python relevant
import numpy as np
import math
# ros-relevant
import rospy
# custom classes
from rl_agent.env_wrapper.ros_env import RosEnvAbs
from sensor_msgs.msg import LaserScan

class RosEnvRawScanPrepWp(RosEnvAbs):
    '''
    This class is a simulation environment wrapper for
    the Polar Representation.
    '''
    def __init__(self, ns, state_collector, execution_mode, task_mode, state_size, observation_space, action_size, action_space, debug, goal_radius, wp_radius, robot_radius, reward_fnc):
        super(RosEnvRawScanPrepWp, self).__init__(ns, state_collector, execution_mode, task_mode, state_size, observation_space, action_size, action_space, debug, goal_radius, wp_radius, robot_radius, reward_fnc)
        self.__res = rospy.get_param("%s/rl_agent/resolution"%ns)

    def get_observation_(self):
        """
        Function returns state that will be fed to the rl-agent
        It includes
        the raw laser scan data,
        the waypoint data in with the same format as the laser scan data.
        The distance of the waypoint is saved
        at the appropriate angle position in the vector.
        :return: state
        """
        waypoint = self.wp_
        num_of_wps = len(waypoint.points)

        state = np.ones(self.STATE_SIZE, dtype=np.float)

        # add laserscan
        state[0, :, 0] = self.merged_scan_.ranges

        # generate wp-vector
        wp_vector = np.zeros(self.STATE_SIZE[1])
        for i in range(num_of_wps):
            dist = math.sqrt(math.pow(waypoint.points[i].x, 2) + math.pow(waypoint.points[i].y, 2))
            angle = math.atan2(waypoint.points[i].y, waypoint.points[i].x) + math.pi
            wp_vector[int(angle/self.merged_scan_.angle_increment)] = dist
        state[0,:,1] = wp_vector

        # Discretize to a resolution of 5cm.
        state = np.round(np.divide(state, self.__res))*self.__res
        if self.debug_:
            debug_scan = LaserScan()
            debug_scan.header = self.merged_scan_.header
            debug_scan.angle_min = self.merged_scan_.angle_min
            debug_scan.angle_max = self.merged_scan_.angle_max
            debug_scan.angle_increment = self.merged_scan_.angle_increment
            debug_scan.range_max = 7.0
            debug_scan.ranges = state[0, :, 0]
            self.debugger_.show_input_scan(debug_scan)
        return state
