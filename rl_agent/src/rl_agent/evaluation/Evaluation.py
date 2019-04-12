'''
    @name:      Analysis_eval.py
    @brief:     The class records relevant data of the agent during driving. The data
                can be later used for analysing the training process.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''

import rospkg
import rospy
import numpy as np
from rl_agent.env_utils.task_generator import TaskGenerator
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import pickle
import time
import math
from geometry_msgs.msg import PoseStamped, Twist, Point
from flatland_msgs.msg import DebugTopicList
from rosgraph_msgs.msg import Clock
from pedsim_msgs.msg import AgentStates
from visualization_msgs.msg import MarkerArray, Marker

class Evaluation():
    '''
    The class records relevant data of the agent during driving. The data
    can be later used for analyising the training process.
    '''
    def __init__(self, state_collector, ns, robot_radius = 0,robot_width = 0.58, robot_height = 0.89):
        self.__robot_radius = robot_radius                              # robot radius
        self.__robot_height = robot_height                              # robot width
        self.__robot_width = robot_width                                # robot heigth
        self.__odom = Odometry()                                        # most recently published odometry of the robot
        self.__path = Path()                                            # most recently published path
        self.__done = False                                             # is episode done?
        self.__new_task_started = False                                 # has a new task started?
        self.__state_collector = state_collector                        # for getting relevant state values of the robot
        self.__rl_agent_path = rospkg.RosPack().get_path('rl_agent')    # absolute path rl_agent-package
        self.__flatland_topics = []                                     # list of flatland topics
        self.__timestep = 0                                             # actual timestemp of training
        self.__NS = ns
        self.MODE = rospy.get_param("%s/rl_agent/train_mode", 1)
        self.__clock = Clock().clock
        self.__task_generator = TaskGenerator(self.__NS, self.__state_collector, 0.46)
        self.__recent_agent_states = []

        # Subscriber for getting data
        self.__odom_sub = rospy.Subscriber("%s/odom"%self.__NS, Odometry, self.__odom_callback, queue_size=1)
        self.__global_plan_sub = rospy.Subscriber("%s/move_base/NavfnROS/plan"%self.__NS, Path, self.__path_callback, queue_size=1)
        self.__done_sub = rospy.Subscriber("%s/rl_agent/done"%self.__NS, Bool, self.__done_callback, queue_size=1)
        self.__new_task_sub = rospy.Subscriber('%s/rl_agent/new_task_started'%self.__NS, Bool, self.__new_task_callback, queue_size=1)
        self.__flatland_topics_sub = rospy.Subscriber("%s/flatland_server/debug/topics"%self.__NS, DebugTopicList, self.__flatland_topic_callback, queue_size=1)
        self.__agent_action_sub = rospy.Subscriber('%s/rl_agent/action'%self.__NS, Twist, self.__trigger_callback)
        self.__ped_sub = rospy.Subscriber('%s/pedsim_simulator/simulated_agents' % self.__NS, AgentStates, self.__ped_callback)
        if self.MODE == 1 or self.MODE == 0:
            self.clock_sub = rospy.Subscriber('%s/clock' % self.__NS, Clock, self.__clock_callback)

        # Publisher for generating qualitative image
        self.__driven_path_pub = rospy.Publisher('%s/rl_eval/driven_path'%self.__NS, Path, queue_size=1)
        self.__driven_path_pub2 = rospy.Publisher('%s/rl_eval/driven_path2'%self.__NS, Path, queue_size=1)
        self.__global_path_pub = rospy.Publisher('%s/rl_eval/global_path'%self.__NS, Path, queue_size=1)
        self.__agent_pub = rospy.Publisher('%s/rl_eval/viz_agents'%self.__NS, MarkerArray, queue_size=1)

    def load_evaluation_set(self, eval_set_path):
        """
        Loading evaluation set with name eval_set_name.
        :param eval_data_path: path to evaluation data
        :param eval_set_name: name of evaluation set (.pickle)
        :return: evaluation_set as list
        """
        with open('%s.pickle'%(eval_set_path), 'rb') as handle:
            self.__evaluation_set = pickle.load(handle)
        return len(self.__evaluation_set)

    def evaluate_set(self, evaluation_set_path, save_path):
        """
        Evaluates an agent with a evaluation_set and saves results in evaluation_data/test
        :param eval_data_path: path to evaluation data
        :param evaluation_set_name: name of evaluation set (.pickle)
        :param agent_name: name of the agent (.pkl)
        """
        task_generator = TaskGenerator(self.__NS, self.__state_collector, 0.46)
        self.load_evaluation_set(evaluation_set_path)
        results = []
        i = 0
        for task in self.__evaluation_set:
            print("Evaluating task %d..."%i)
            task_generator.set_task(task)
            self.__done = False
            result = self.evaluate_episode(False)
            results.append(result)
            i += 1
            with open('%s.pickle' % (save_path), 'ab') as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate_episode(self, train):
        """
        Evaluates current episode
        :return result of the episode.

        """
        result = {"global_path": self.__path, "agent_states": []}
        done = False
        max_time = len(self.__path.poses)/10*2  # in secs
        if not self.MODE == 2:
            start_time = self.__clock
        else:
            start_time = rospy.get_rostime()
        driven_route = Path()
        driven_route.header = self.__path.header
        poses = []
        while not done:
            [static_scan, ped_scan, merged_scan, img, wp, twist, goal] = self.__state_collector.get_state()
            min_obstacle_dist = np.amin(merged_scan.ranges)
            dist_to_goal = math.sqrt(math.pow(goal.position.x, 2) + math.pow(goal.position.y, 2))

            # Check if task over
            pose = PoseStamped()
            pose.header = self.__odom.header
            pose.pose = self.__odom.pose.pose
            poses.append(pose)
            if not self.MODE == 2:
                now = self.__clock
            else:
                now = rospy.get_rostime()
            # Check if task over
            if min_obstacle_dist <= self.__robot_radius or \
                self.__rect_robot_collision(static_scan, ped_scan, self.__robot_width, self.__robot_height):
                rospy.loginfo("Robot collided with obstacle.")
                done = True
                result["success"] = -1
            elif dist_to_goal < 0.65:
                rospy.loginfo("Goal reached.")
                done = True
                result["success"] = 1
            elif self.__done or (now - start_time).to_sec() > max_time:
                rospy.loginfo("Time exceeded.")
                done = True
                result["success"] = 0
            if (not train):
                result["agent_states"].append(self.__recent_agent_states)
            self.__sleep(0.1)
        result["num_stat_obj"] = 0
        result["num_peds"] = 0

        #Counting number of static objects and number of dynamic objects (pedestrians)
        for topic in self.__flatland_topics:
            if topic.find("stat_obj") != -1:
                result["num_stat_obj"] +=1
                continue
            if topic.find("person") != -1:
                result["num_peds"] +=1

        driven_route.poses = poses
        result["time"] = self.__clock - start_time
        result["driven_path"] = driven_route
        result["timestep"] = self.__timestep
        return result

    def __sleep(self, secs):
        """
        Sleep method, that takes into account the namespace we are in.
        As we don't run the python3.5-scripts as rosnodes, manual redirecting necessary.
        :return result of the episode.

        """
        if not self.MODE == 2:
            now = self.__clock
            while (self.__clock - now).to_sec() < secs:
                time.sleep(0.0001)
        else:
            # time.sleep(secs)
            rospy.sleep(secs)

    def evaluate_training(self, save_path):
        """
        Evaluates an agent during training. Results are saved in <evaluation_data_path>/evaluation_data/train
        :param agent_name: name of the agent (.pkl)
        """

        while True:
            self.__timestep -= 1

            #Waiting for new task
            while not self.__new_task_started:
                time.sleep(0.001)
            self.__done = False
            self.__new_task_started = False
            result = self.evaluate_episode(True)
            with open('%s.pickle' % (save_path), 'ab') as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def show_paths(self, result):
        """
        Publishs the driven and global path of the result
        :param result
        """
        self.__driven_path_pub.publish(result["driven_path"])
        self.__global_path_pub.publish(result["global_path"])

    def generate_qualitative_static_image_rviz(self, results, results_comp, i_task, i_pos):
        """
        Generates a qualitative image of the result of an evaluation set. Furthermore a second agent
        can be plotted to compare both in the image (generated in rviz)
        :param evaluation_set_name: name of the used evaluation set
        :param results: results of first agent
        :param results_comp: results of second agent
        :param i_task: index of episode in evaluation set that should be displayed
        :param i_pos: position where robot should be displayed.
        """
        if len(results) != len(self.__evaluation_set):
            print("Error: results and evaluations_set need to have the same length. They don't fit.")
            return
        self.__global_path_pub.publish(results[i_task]["global_path"])
        self.__driven_path_pub.publish(results[i_task]["driven_path"])
        if len(results) == len(results_comp):
            self.__driven_path_pub2.publish(results_comp[i_task]["driven_path"])

        x_robot = results[i_task]["driven_path"].poses[i_pos].pose.position.y
        y_robot = results[i_task]["driven_path"].poses[i_pos].pose.position.x
        theta = math.atan2(results[i_task]["driven_path"].poses[i_pos + 4].pose.position.y - x_robot,
                           results[i_task]["driven_path"].poses[i_pos + 4].pose.position.x - y_robot)
        self.__task_generator.set_robot_pos(y_robot, x_robot, theta)
        self.__task_generator.remove_all_static_objects()
        task = self.__evaluation_set[i_task]

        if 'static_objects' in task.keys():
            for i in range(len(task["static_objects"]["x"])):
                self.__task_generator.spawn_object(task["static_objects"]["model_name"][i], i, task["static_objects"]["x"][i],
                                    task["static_objects"]["y"][i], task["static_objects"]["theta"][i])

        # Making visible by taking a sim_step
        self.__task_generator.take_sim_step()

    def generate_qualitative_ped_image_rviz(self, results, i_task, i_pos):
        """
           Generates a dynamic qualitative image of the result of a pedestrian evaluation set.
           The state of the pedestrians and the robot is plotted at position i_pos.
           :param evaluation_set_name: name of the used evaluation set
           :param results: results of first agent
           :param results_comp: results of second agent
           :param i_task: index of episode in evaluation set that should be displayed
           :param i_pos: position where robot should be displayed.
           """

        if (i_pos >= len(results[i_task]["success"])-4):
            i_pos = len(results[i_task]["success"])-5
        self.generate_qualitative_static_image_rviz(results, [], i_task, i_pos)
        agent_states = results[i_task]["agent_states"][i_pos]
        marker_array = MarkerArray()
        for i, agent_state in enumerate(agent_states):
            marker_array.markers.append(Marker())
            marker_array.markers[i*3] = Marker()
            marker_array.markers[i*3].header.frame_id = "/map"
            marker_array.markers[i*3].id = i*3
            marker_array.markers[i*3].action = 0
            marker_array.markers[i*3].type = 0
            marker_array.markers[i*3].scale.x = 0.07
            marker_array.markers[i*3].scale.y = 0.2
            marker_array.markers[i*3].scale.z = 0.3
            marker_array.markers[i*3].color.a = 0.5
            marker_array.markers[i*3].color.r = 170/255
            marker_array.markers[i*3].color.g = 0.0
            marker_array.markers[i*3].color.b = 0.0

            start_point = agent_state.pose.position
            marker_array.markers[i*3].points.append(start_point)
            end_point = Point()
            fac = 3
            end_point.x = agent_state.pose.position.x + agent_state.twist.linear.x*fac
            end_point.y = agent_state.pose.position.y + agent_state.twist.linear.y*fac
            marker_array.markers[i*3].points.append(end_point)

            marker_array.markers.append(Marker())
            marker_array.markers[i*3+1].header.frame_id = "/map"
            marker_array.markers[i*3+1].id = i*3+1
            marker_array.markers[i*3+1].action = 0
            marker_array.markers[i*3+1].type = 2
            marker_array.markers[i*3+1].pose = agent_state.pose
            marker_array.markers[i*3+1].scale.x = 0.32
            marker_array.markers[i*3+1].scale.y = 0.32
            marker_array.markers[i*3+1].scale.z = 0.1
            marker_array.markers[i*3+1].color.a = 1.0
            marker_array.markers[i*3+1].color.r = 170/255
            marker_array.markers[i*3+1].color.g = 0.0
            marker_array.markers[i*3+1].color.b = 0.0

            lookahead = 100
            marker_array.markers.append(Marker())
            marker_array.markers[i*3+2].header.frame_id = "/map"
            marker_array.markers[i*3+2].id = i*3+2
            marker_array.markers[i*3+2].action = 0
            marker_array.markers[i*3+2].type = 4
            marker_array.markers[i*3+2].scale.x = 0.015
            marker_array.markers[i*3+2].color.a = 1.0
            marker_array.markers[i*3+2].color.r = 170/255
            marker_array.markers[i*3+2].color.g = 0.0
            marker_array.markers[i*3+2].color.b = 0.0
            for agent_states_lookahead in results[i_task]["agent_states"][i_pos:min(len(results[i_task]["agent_states"]), i_pos + lookahead)]:
                marker_array.markers[i*3+2].points.append(agent_states_lookahead[i].pose.position)
            # Publish Marker
        self.__agent_pub.publish(marker_array)

    def __rect_robot_collision(self, static_scan_msg, ped_scan_msg, robot_width, robot_height):
        """
        Checks if the robot footprint robot_width x robot_height collided
        :param static_scan_msg: scan of the static objects
        :param ped_scan_msg: scan of the pedestrians
        :param robot_width: width of the robots footprint
        :param robot_height: height of the robots footprint
        """
        robot_collision = False
        angle_min = ped_scan_msg.angle_min
        angle_increment = ped_scan_msg.angle_increment
        for i in range(len(ped_scan_msg.ranges)):
            angle = angle_min + i * angle_increment
            length = min(ped_scan_msg.ranges[i], static_scan_msg.ranges[i])
            x = math.cos(angle) * length
            y = math.sin(angle) * length
            if(x > -robot_height/2 and x < robot_height/2 and y > -robot_width/2 and y < robot_width/2):
                robot_collision = True

        return robot_collision

    def __odom_callback(self, data):
        self.__odom = data

    def __path_callback(self, data):
        self.__path = data

    def __done_callback(self, data):
        self.__done = data.data

    def __new_task_callback(self, data):
        self.__new_task_started = data.data

    def __flatland_topic_callback(self, data):
        self.__flatland_topics = data.topics

    def __trigger_callback(self, data):
        self.__timestep += 1

    def __clock_callback(self, data):
        self.__clock = data.clock

    def __ped_callback(self, data):
        self.__recent_agent_states = data.agent_states

