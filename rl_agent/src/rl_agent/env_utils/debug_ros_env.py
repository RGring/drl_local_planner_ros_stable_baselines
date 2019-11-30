'''
    @name:      debug_ros_env.py
    @brief:     This class provides debugging methods RL-relevant data.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''

#ros-relevant
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import LaserScan
from collections import deque

#python-relevant
import numpy as np


class DebugRosEnv():
    """
        This class serves as debugger for RL-relevant data like:
        - input state
        - rewards
        """
    def __init__(self, ns, stack_offset):
        self.__ns = ns
        self.__stack_offset = stack_offset
        print("stack_offset: %d"%self.__stack_offset)
        self.__input_images = deque(maxlen=4 * self.__stack_offset)

        #Input state
        self.__input_img_pub1 = rospy.Publisher('%s/state_image1' % (self.__ns), Image, queue_size=1)
        self.__input_img_pub2 = rospy.Publisher('%s/state_image2' % (self.__ns), Image, queue_size=1)
        self.__input_img_pub3 = rospy.Publisher('%s/state_image3' % (self.__ns), Image, queue_size=1)
        self.__input_img_pub4 = rospy.Publisher('%s/state_image4' % (self.__ns), Image, queue_size=1)
        self.__occ_grid_pub = rospy.Publisher('%s/rl_map' % (self.__ns), OccupancyGrid, queue_size=1)
        self.__input_scan_pub = rospy.Publisher('%s/state_scan' % (self.__ns), LaserScan, queue_size=1)

        # reward info
        self.__rew_pub = rospy.Publisher('%s/reward' % (self.__ns), Marker, queue_size=1)
        self.__rew_num_pub = rospy.Publisher('%s/reward_num' % (self.__ns), Float64, queue_size=1)

        # Waypoint info
        self.__wp_pub1 = rospy.Publisher('%s/wp_vis1' % (self.__ns), PointStamped, queue_size=1)
        self.__wp_pub2 = rospy.Publisher('%s/wp_vis2' % (self.__ns), PointStamped, queue_size=1)
        self.__wp_pub3 = rospy.Publisher('%s/wp_vis3' % (self.__ns), PointStamped, queue_size=1)
        self.__wp_pub4 = rospy.Publisher('%s/wp_vis4' % (self.__ns), PointStamped, queue_size=1)



    def show_wp(self, data):
        """
        Publishing waypoints on the path (maximum 4).
        :param data: waypoint message
        """
        msg = PointStamped()
        msg.header = data.header
        msg.point = data.points[0]
        self.__wp_pub1.publish(msg)
        if(len(data.points) > 1):
            msg.point = data.points[1]
            self.__wp_pub2.publish(msg)
        if (len(data.points) > 2):
            msg.point = data.points[2]
            self.__wp_pub3.publish(msg)
        if (len(data.points) > 3):
            msg.point = data.points[3]
            self.__wp_pub4.publish(msg)

    def show_image_stack(self, data):
        """
        Publishing input image stack. Maximal 4 images are displayed in rviz.
        :param data: input matrix
        """
        self.__input_images.appendleft(data)
        self.__input_img_pub1.publish(self.__data_to_image(self.__input_images[0]))
        if(len(self.__input_images) > self.__stack_offset):
            self.__input_img_pub2.publish(self.__data_to_image(self.__input_images[self.__stack_offset]))
        if (len(self.__input_images) > 2*self.__stack_offset):
            self.__input_img_pub3.publish(self.__data_to_image(self.__input_images[self.__stack_offset * 2]))
        if (len(self.__input_images) > 3*self.__stack_offset):
            self.__input_img_pub4.publish(self.__data_to_image(self.__input_images[self.__stack_offset * 3]))

    def __data_to_image(self, data):
        """
        Transforms input state format to Image msg, displayable in rviz.
        :param data: input state
        """
        msg = Image()
        msg.header.frame_id = "/base_footprint"
        msg.height = data.shape[1]
        msg.width = data.shape[0]
        msg.encoding = "mono8"
        msg.data = np.uint8(np.ndarray.flatten(data, order='F'))[::-1].tolist()
        return msg

    def show_input_image(self, data):
        """
        Publishing single input matrix as image.
        Can be displayed in rviz.
        :param data: input matrix
        """
        msg = self.__data_to_image(data)
        self.__input_img_pub1.publish(msg)

    def show_input_scan(self, scan):
        """
        Publishing input scan for displaying in rviz
        :param data: input matrix
        """
        self.__input_scan_pub.publish(scan)

    def show_input_occ_grid(self, state_image):
        """
        Publishing occupancy grid.
        :param state_image: occupancy grid
        """
        self.__occ_grid_pub.publish(state_image)

    def show_reward(self, reward):
        """
        Publishing reward value as marker.
        :param reward
        """
        # Publish reward as Marker
        msg = Marker()
        msg.header.frame_id = "/base_footprint"
        msg.ns = ""
        msg.id = 0
        msg.type = msg.TEXT_VIEW_FACING
        msg.action = msg.ADD

        msg.pose.position.x = 0.0
        msg.pose.position.y = 1.0
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        msg.text = "%f"%reward

        msg.scale.x = 10.0
        msg.scale.y = 10.0
        msg.scale.z = 1.0

        msg.color.r = 0.3
        msg.color.g = 0.4
        msg.color.b = 1.0
        msg.color.a = 1.0
        self.__rew_pub.publish(msg)

        # Publish reward as number
        msg = Float64()
        msg.data = reward
        self. __rew_num_pub.publish(msg)

