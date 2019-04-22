'''
    @name:      task_generator.py
    @brief:     This class generates random tasks for training.
    @author:    Ronja Güldenring
    @version:   3.5
    @date:      2019/04/05
'''

# python-relevant
import time
import math
import random
from pyquaternion import Quaternion
import numpy as np
from collections import deque
import pickle

#ros-relevant
import rospy
import rospkg

from nav_msgs.msg import OccupancyGrid, Path
from flatland_msgs.srv import MoveModel, DeleteModel, SpawnModel, Step
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose2D
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Twist, Point
from pedsim_srvs.srv import SpawnPed
from std_srvs.srv import SetBool, Empty

class TaskGenerator():
    """
    This class generates task for training. The following task generation is implemented:
    - generate random paths with static objects on it.
    - generate random paths with pedestrians on it.
    - generate random paths with pedestrians and static objects on it.
    - load predefined path (--> evaluation).
    """
    def __init__(self, ns, state_collector, robot_radius):

        # Class variables
        self.NS = ns                                    # namespace
        self.ROBOT_RADIUS = robot_radius                # radius of the robot
        self.__update_rate = \
            1/rospy.get_param("%s/rl_agent/update_frequency"%ns)
        self.__state_collector = state_collector        # Collects state information
        self.__move_base_status_id = ""                 # recent id of move_base status
        self.__map = OccupancyGrid()                    # global map
        self.__path = Path()                            # global plan
        self.__static_objects = {"name": [],
                                 "model_name": [],
                                 "x": [],
                                 "y": [],
                                 "theta": []}           # static objects that has been spawned

        self.__ped_type = 11                            # 0: Pedestrians don't avoid robot
                                                        # 10: Pedestrians always avoid robot
                                                        # 11: Pedestrians avoid robot if it stands still and after reaction time.

        self.__peds = []                                # pedestrians that has been spawned

        self.__old_path_stamp = 0.0                     # timestamp of last global plan
        self.init = True
        self.__flatland_path = rospkg.RosPack().get_path('flatland_setup')
        self.__static_object_types = ["cylinder.model.yaml",  # different object types
                        "palett.model.yaml",
                        "wagon.model.yaml"]

        # Services
        self.__sim_step = rospy.ServiceProxy('%s/step' % self.NS, Step)
        self.__sim_pause = rospy.ServiceProxy('%s/pause' % self.NS, Empty)
        self.__sim_resume = rospy.ServiceProxy('%s/resume' % self.NS, Empty)
        self.__move_robot_to = rospy.ServiceProxy('%s/move_model' % self.NS, MoveModel)
        self.__delete_model = rospy.ServiceProxy('%s/delete_model' % self.NS, DeleteModel)
        self.__spawn_model = rospy.ServiceProxy('%s/spawn_model' % self.NS, SpawnModel)
        self.__spawn_ped_srv = rospy.ServiceProxy('%s/pedsim_simulator/spawn_ped' % self.NS, SpawnPed)
        self.__remove_all_peds_srv = rospy.ServiceProxy('%s/pedsim_simulator/remove_all_peds' % self.NS, SetBool)

        # Subscriber
        self.__goal_status_sub = rospy.Subscriber("%s/move_base/status" % self.NS, GoalStatusArray,
                                                  self.__goal_status_callback, queue_size=1)
        self.__map_sub = rospy.Subscriber("%s/map" % self.NS, OccupancyGrid, self.__map_callback)
        self.__path_sub = rospy.Subscriber("%s/move_base/NavfnROS/plan" % self.NS, Path, self.__path_callback)

        # Publisher
        self.__initialpose_pub = rospy.Publisher('%s/initialpose' % self.NS, PoseWithCovarianceStamped, queue_size=1)
        self.__goal_pub_ = rospy.Publisher('%s/move_base_simple/goal' % self.NS, PoseStamped, queue_size=1)
        self.__cmd_vel_pub = rospy.Publisher('%s/cmd_vel' % self.NS, Twist, queue_size=1)
        self.__done_pub = rospy.Publisher('%s/rl_agent/done' % self.NS, Bool, queue_size=1)
        self.__new_task_pub = rospy.Publisher('%s/rl_agent/new_task_started' % self.NS, Bool, queue_size=1)
        self.__resume = rospy.ServiceProxy('%s/resume' % (self.NS), Empty)


    def set_task(self, task):
        """
        Loading predefined task (e.g. during evaluation)
        :param task predefined task, that will be loaded
        :return True, if task is loaded successfully
        """
        is_path_available = False
        count = 0
        while not is_path_available and count < 10:
            self.__spread_done()
            self.__stop_robot()
            if not task:
                return False

            # Setting path
            self.set_robot_pos(task["start"][0], task["start"][1], task["start"][2])
            self.take_sim_step()
            self.__wait_for_laser()
            time.sleep(0.11)
            self.__publish_goal(task["goal"][0], task["goal"][1], task["goal"][2])
            is_path_available = self.__is_new_path_available(task["goal"], task["start"])
            print("path not available...trying again...")
            count+=1

        # Spawning obstacles
        self.remove_all_static_objects()
        self.__remove_all_peds()

        if 'static_objects' in task.keys():
            for i in range(len(task["static_objects"]["x"])):
                self.spawn_object(task["static_objects"]["model_name"][i], i, task["static_objects"]["x"][i], task["static_objects"]["y"][i], task["static_objects"]["theta"][i])

        if 'peds' in task.keys():
            id = 1
            for ped in task["peds"]:
                self.__spawn_ped(ped["start_pos"], ped["wps"], id)
                id +=1
        self.__spread_new_task()
        return True

    def set_path(self):
        """
        Generating a random path in environment.
        :return d task information containing start, goal and path
        """
        self.__spread_done()
        d = {}
        d["start"] = self.__set_random_robot_pos()
        d["goal"] = self.__publish_random_goal_()
        self.__is_new_path_available(d["goal"], d["start"])
        d["path"] = self.__path
        return d

    def set_random_static_task(self):
        """
        Generating a random path with static obstacles on it.
        :return d task information containing start, goal, path and static objects
        """
        self.__spread_done()
        d = {}
        # Setting path
        self.__stop_robot()
        d["start"] = self.__set_random_robot_pos()
        d["goal"] = self.__publish_random_goal_()

        # Spawning new obstacles
        self.remove_all_static_objects()
        self.__remove_all_peds()
        if self.__is_new_path_available(d["goal"], d["start"]):
            self.__spawn_random_static_objects()
        d["static_objects"] = self.__static_objects
        d["path"] = self.__path
        self.__spread_new_task()
        return d

    def set_random_ped_task(self):
        """
        Generating a random path with pedestrians on it.
        :return d task information containing start, goal, path and pedestrians
        """
        self.__spread_done()
        d = {}
        self.__stop_robot()
        d["start"] = self.__set_random_robot_pos()
        d["goal"] = self.__publish_random_goal_()
        self.__remove_all_peds()
        self.remove_all_static_objects()
        if self.__is_new_path_available(d["goal"], d["start"]):
            self.__spawn_random_peds_on_path()
        d["peds"] = self.__peds
        d["path"] = self.__path
        self.__spread_new_task()
        return d


    def set_random_static_ped_task(self):
        """
        Generating a random path with pedestrians and static obstacles on it.
        :return d task information containing start, goal, path, pedestrians and static objects
        """
        # Setting path
        self.__spread_done()
        d = {}
        self.__stop_robot()
        d["start"] = self.__set_random_robot_pos()
        d["goal"] = self.__publish_random_goal_()

        # Spawning new obstacles
        self.__remove_all_peds()
        self.remove_all_static_objects()
        if self.__is_new_path_available(d["goal"], d["start"]):
            self.__spawn_random_static_objects()
            self.__spawn_random_peds_on_path()
        d["peds"] = self.__peds
        d["path"] = self.__path
        d["static_objects"] = self.__static_objects
        self.__spread_new_task()
        return d

    def take_sim_step(self):
        """
        Executing one simulation step of 0.1 sec
        """
        msg = Float64()
        msg.data = self.__update_rate
        rospy.wait_for_service('%s/step' % self.NS)
        self.__sim_step(msg)
        return

    def __set_random_robot_pos(self):
        """
        Moving robot to random position (x, y, theta) in simulation
        :return: robot position [x, y, theta]
        """
        success = False
        while not success:
            x, y, theta = self.__get_random_pos_on_map(self.__map)
            self.set_robot_pos(x, y, theta)
            scan = self.__wait_for_laser()
            min_obstacle_dist = np.amin(scan)
            if min_obstacle_dist > (self.ROBOT_RADIUS + 0.1):
                success = True
        return x, y, theta

    def set_robot_pos(self, x, y, theta):
        """
        Move robot to position (x, y, theta) in simulation
        :param x x-position of the robot
        :param y y-position of the robot
        :param theta theta-position of the robot
        """
        pose = Pose2D()
        pose.x = x
        pose.y = y
        pose.theta = theta
        rospy.wait_for_service('%s/move_model' % self.NS)
        self.__move_robot_to('robot_1', pose)
        self.take_sim_step()
        self.__pub_initial_position(x, y, theta)
        self.take_sim_step()

    def __wait_for_laser(self):
        """
        Waiting for most recent laserscan to get available
        :return most recent laser scan data
        """
        ts = self.__state_collector.get_static_scan().header.stamp
        self.take_sim_step()
        scan = self.__state_collector.get_static_scan()
        begin = time.time()
        while len(scan.ranges) == 0 or scan.header.stamp <= ts:
            rospy.logdebug("Waiting for laser scan to get available.")
            if(time.time() - begin > 1):
                self.take_sim_step()
            time.sleep(0.01)
            scan = self.__state_collector.get_static_scan()
        return scan.ranges

    def __pub_initial_position(self, x, y, theta):
        """
        Publishing new initial position (x, y, theta) --> for localization
        :param x x-position of the robot
        :param y y-position of the robot
        :param theta theta-position of the robot
        """
        initpose = PoseWithCovarianceStamped()
        initpose.header.stamp = rospy.get_rostime()
        initpose.header.frame_id = "map"
        initpose.pose.pose.position.x = x
        initpose.pose.pose.position.y = y
        quaternion = self.__yaw_to_quat(theta)

        initpose.pose.pose.orientation.w = quaternion[0]
        initpose.pose.pose.orientation.x = quaternion[1]
        initpose.pose.pose.orientation.y = quaternion[2]
        initpose.pose.pose.orientation.z = quaternion[3]
        self.__initialpose_pub.publish(initpose)
        return

    def __publish_random_goal_(self):
        """
        Publishing new random goal [x, y, theta] for global planner
        :return: goal position [x, y, theta]
        """
        x, y, theta = self.__get_random_pos_on_map(self.__map)
        self.__publish_goal(x, y, theta)
        return x, y, theta

    def __publish_goal(self, x, y, theta):
        """
        Publishing goal (x, y, theta)
        :param x x-position of the goal
        :param y y-position of the goal
        :param theta theta-position of the goal
        """
        self.__old_path_stamp = self.__path.header.stamp
        goal = PoseStamped()
        goal.header.stamp = rospy.get_rostime()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        quaternion = self.__yaw_to_quat(theta)
        goal.pose.orientation.w = quaternion[0]
        goal.pose.orientation.x = quaternion[1]
        goal.pose.orientation.y = quaternion[2]
        goal.pose.orientation.z = quaternion[3]
        self.__goal_pub_.publish(goal)
        return

    def __get_random_pos_on_map(self, map):
        """
        Find a valid (free) random position (x, y, theta) on the map
        :param map
        :return: x, y, theta
        """
        map_width = map.info.width * map.info.resolution + map.info.origin.position.x
        map_height = map.info.height * map.info.resolution + map.info.origin.position.y
        x = random.uniform(0.0 , map_width)
        y = random.uniform(0.0, map_height)
        while not self.__is_pos_valid(x, y, map):
            x = random.uniform(0.0, map_width)
            y = random.uniform(0.0, map_height)

        theta = random.uniform(-math.pi, math.pi)
        return x, y, theta

    def __is_pos_valid(self, x, y, map):
        """
        Checks if position (x,y) is a valid position on the map.
        :param  x x-position
        :param  y y-position
        :param  map
        :return: True if position is valid
        """
        cell_radius = int((self.ROBOT_RADIUS + 0.1)/map.info.resolution)
        y_index =  int((y-map.info.origin.position.y)/map.info.resolution)
        x_index =  int((x-map.info.origin.position.x)/map.info.resolution)

        for i in range(x_index-cell_radius, x_index+cell_radius, 1):
            for j in range(y_index-cell_radius, y_index+cell_radius, 1):
                index = j * map.info.width + i
                if index >= len(map.data):
                    return False
                val = map.data[index]
                if val != 0:
                    return False
        return True

    def __spread_done(self):
        """
        Pulishing True on done-pub
        """
        self.__done_pub.publish(Bool(True))

    def __spread_new_task(self):
        """
        Pulishing True if new task is set. Ready to be solved
        """
        self.__new_task_pub.publish(Bool(True))

    def __stop_robot(self):
        """
        Stops robot.
        """
        # self.__agent_action_pub.publish(Twist())
        self.__cmd_vel_pub.publish(Twist())
        return

    def __spawn_random_static_objects(self):
        """
        Spawns a random number of static objects randomly on the path.
        """
        max_num_obstacles = int(len(self.__path.poses) / 170)
        if max_num_obstacles == 0:
            num_static_obstacles = 0
        else:
            num_static_obstacles = random.randint(1, max_num_obstacles)
        for i in range(num_static_obstacles):
            model_name = random.choice(self.__static_object_types)
            [x, y] = self.__generate_rand_pos_on_path(self.__path.poses, 100, 1.0)
            theta = random.uniform(-math.pi, math.pi)
            self.spawn_object(model_name, i, x, y, theta)
        return

    def spawn_object(self, model_name, index, x, y, theta):
        """
        Spawns one static object.
        :param  model_name object type
        :param  x x-position of the object
        :param  y y-position of the object
        :param  theta orientation of the object
        """
        srv = SpawnModel()
        srv.yaml_path = "%s/objects/%s" % (self.__flatland_path, model_name)
        srv.name = "stat_obj_%d" % index
        srv.ns = "stat_obj_%d" % index
        temp = Pose2D()
        temp.x = x
        temp.y = y
        temp.theta = theta
        srv.pose = temp
        rospy.wait_for_service('%s/spawn_model' % self.NS)
        try:
            self.__spawn_model(srv.yaml_path, srv.name, srv.ns, srv.pose)
        except (TypeError):
            print('Spawn object: TypeError.')
            return
        except (rospy.ServiceException):
            print('Spawn object: rospy.ServiceException. Closing serivce')
            try:
                self.__spawn_model.close()
            except AttributeError:
                print('Spawn object close(): AttributeError.')
                return
            return
        except AttributeError:
            print('Spawn object: AttributeError.')
            return

        self.__static_objects["name"].append(srv.name)
        self.__static_objects["model_name"].append(model_name)
        self.__static_objects["x"].append(x)
        self.__static_objects["y"].append(y)
        self.__static_objects["theta"].append(theta)

        return

    def remove_all_static_objects(self):
        """
         Removes all static objects, that has been spawned so far
         """
        for i in self.__static_objects["name"]:
            srv = DeleteModel()
            srv.name = i
            rospy.wait_for_service('%s/delete_model' % self.NS)
            ret = self.__delete_model(srv.name)

        # This is necessary in case old objects are still present in flatland
        if self.__static_objects["name"] == []:
            for i in range(10):
                srv = DeleteModel()
                srv.name = "stat_obj_%d"%(i)
                rospy.wait_for_service('%s/delete_model' % self.NS)
                ret = self.__delete_model(srv.name)
                if not ret.success:
                    break;
        self.__static_objects = {"name": [], "model_name": [], "x": [], "y": [], "theta": []}

    def __remove_all_peds(self):
        """
        Removes all pedestrians, that has been spawned so far
        """
        srv = SetBool()
        srv.data = True
        rospy.wait_for_service('%s/pedsim_simulator/remove_all_peds' % self.NS)
        self.__remove_all_peds_srv(srv.data)
        self.__peds = []
        return

    def spawn_random_peds_in_world(self, n):
        """
        Spawning n random pedestrians in the whole world.
        :param n number of pedestrians that will be spawned.
        """
        for i in range(1, n):
            waypoints = np.array([], dtype=np.int64).reshape(0, 3)
            [x, y, theta] = self.__get_random_pos_on_map(self.__map)
            waypoints = np.vstack([waypoints, [x, y, 0.3]])
            if random.uniform(0.0, 1.0) < 0.8:
                for j in range(4):
                    dist = 0
                    while dist < 4:
                        [x2, y2, theta2] = self.__get_random_pos_on_map(self.__map)
                        dist = self.__mean_sqare_dist_((waypoints[-1,0] - x2), (waypoints[-1,1] - y2))
                    waypoints = np.vstack([waypoints, [x2, y2, 0.3]])
            self.__spawn_ped([x, y, 0.0], waypoints, i)

    def __spawn_random_peds_on_path(self):
        """
        Spawns a random number of pedestrians randomly near the path.
        Pedestrians either cross the path, walk along the path or stand around.
        """
        max_num_peds = int(len(self.__path.poses)/150)

        if(max_num_peds == 0):
            num_peds = 0
        else:
            num_peds = random.randint(1,max_num_peds)
        id = 1
        # Pedestrians along the path
        for i in range(math.ceil(num_peds*0.8)):
            self.__spawn_random_ped_along_path(id)
            id += 1

        # Pedestrians crossing path
        for i in range(math.ceil(num_peds/2)):
            self.__spawn_random_crossing_ped(id)
            id += 1
        return

    def __spawn_random_ped_along_path(self, id):
        """
         Spawning a pedestrian walking along the path. The pedestrian's waypoints are generated randomply
         """
        start_pos = self.__generate_rand_pos_on_path(self.__path.poses, 100, 2)
        start_pos.append(0.0)

        waypoints = np.array([], dtype=np.int64).reshape(0, 3)
        if random.uniform(0.0, 1.0) < 0.85:
            waypoints = np.vstack([waypoints, [start_pos[0], start_pos[1], 0.3]])

        wp = self.__generate_rand_pos_on_path(self.__path.poses, 100, 2)
        dist = self.__mean_sqare_dist_((wp[0] - start_pos[0]), (wp[1] - start_pos[1]))
        count = 0

        #Trying to find a waypoint with a minimum distance of 5.
        while (dist < 5 and count < 10):
            wp = self.__generate_rand_pos_on_path(self.__path.poses, 100, 2)
            dist = self.__mean_sqare_dist_((wp[0] - start_pos[0]), (wp[1] - start_pos[1]))
            count += 1

        if (count < 10):
            # Found waypoint with a minimum distance of 5m
            wp.append(0.3)
            waypoints = np.vstack([waypoints, wp])
        else:
            # Didn't find waypoint with a minimum distance of 5m --> pedestrian will stand around.
            waypoints = np.vstack([waypoints, [start_pos[0], start_pos[1], 0.3]])
        self.__spawn_ped(start_pos, waypoints, id)

    def __spawn_random_crossing_ped(self, id):
        """
         Spawning a pedestrian crossing the path. The pedestrian's waypoints are generated randomply
         """
        pos_index = random.randint(0, len(self.__path.poses) - 2)
        path_pose = self.__path.poses[pos_index]
        path_pose_temp = self.__path.poses[pos_index + 1]
        angle = math.atan2((path_pose_temp.pose.position.y - path_pose.pose.position.y),
                           (path_pose_temp.pose.position.x - path_pose.pose.position.x)) + math.pi / 2

        start_pos = self.__generate_rand_pos_near_pos(path_pose, 4, angle + math.pi)
        start_pos.append(0.0)

        waypoints = np.array([], dtype=np.int64).reshape(0, 3)
        wp1 = self.__generate_rand_pos_near_pos(path_pose, 4, angle + math.pi)
        waypoints = np.vstack([waypoints, [wp1[0], wp1[1], 0.3]])

        wp2 = self.__generate_rand_pos_near_pos(path_pose, 4, angle)
        dist = self.__mean_sqare_dist_((wp2[0] - wp1[0]), (wp2[1] - wp1[1]))
        count = 0
        #Trying to find a waypoint with a minimum distance of 5.
        while (dist < 5 and count < 10):
            wp2 = self.__generate_rand_pos_near_pos(path_pose, 4, angle)
            dist = self.__mean_sqare_dist_((wp2[0] - wp1[0]), (wp2[1] - wp1[1]))
            count += 1

        if (count < 10):
            # Found waypoint with a minimum distance of 5m
            wp2.append(0.3)
            waypoints = np.vstack([waypoints, wp2])

        rand = random.uniform(0.0, 1.0)
        start_pos = [wp1[0] + rand * (wp2[0] - wp1[0]), wp1[1] + rand * (wp2[1] - wp1[1]), 0.0]
        self.__spawn_ped(start_pos, waypoints, id)

    def __spawn_ped(self, start_pos, wps, id):
        """
        Spawning one pedestrian in the simulation.
        :param  start_pos start position of the pedestrian.
        :param  wps waypoints the pedestrian is supposed to walk to.
        :param  id id of the pedestrian.
        """
        srv = SpawnPed()
        srv.id = id
        temp = Point()
        temp.x = start_pos[0]
        temp.y = start_pos[1]
        temp.z = start_pos[2]
        srv.pos = temp
        srv.type = self.__ped_type
        srv.number_of_peds = 1
        srv.waypoints = []
        for pos in wps:
            p = Point()
            p.x = pos[0]
            p.y = pos[1]
            p.z = pos[2]
            srv.waypoints.append(p)

        rospy.wait_for_service('%s/pedsim_simulator/spawn_ped' % self.NS)
        try:
            self.__spawn_ped_srv(srv.id, srv.pos, srv.type, srv.number_of_peds, srv.waypoints)
        except rospy.ServiceException:
            print('Spawn object: rospy.ServiceException. Closing serivce')
            try:
                self.__spawn_model.close()
            except AttributeError:
                print('Spawn object close(): AttributeError.')
                return

        self.__peds.append({"start_pos": start_pos, "wps": wps})
        return

    def __generate_rand_pos_on_path(self, path, range, max_r):
        """
        Generating a random position on/near a path.
        :param  path the path on that the position is generated
        :param  range the range from the beginning and end of the path,
        that is not considered during generation
        :param  max_r the radius around the path, where the position is generated
        """
        try:
            path_pose = random.choice(path[range:-range])
        except IndexError:
            try:
                path_pose = path[range]
            except IndexError:
                return [0, 0]
        pos_on_map = False
        while not pos_on_map:
            alpha = 2 * math.pi * random.random()
            r = max_r * math.sqrt(random.random())
            x = r * math.cos(alpha) + path_pose.pose.position.x
            y = r * math.sin(alpha) + path_pose.pose.position.y
            pos_on_map = self.__is_pos_valid(x, y, self.__map)
        return [x,y]

    def __is_new_path_available(self, goal, start):
        """
        Waiting for path to be published.
        :return True if path is available

        """
        is_available = False
        begin = time.time()
        while (time.time() - begin) < 20.0:
            if self.__path.header.stamp <= self.__old_path_stamp or len(self.__path.poses) == 0:
                time.sleep(0.01)
            else:
                is_available = True
                break
        self.__old_path_stamp = self.__path.header.stamp
        dist_goal = 10
        dist_start = 10

        if len(self.__path.poses) > 2:
            #Checking of correct path has been published
            dist_goal = self.__mean_sqare_dist_((goal[0] - self.__path.poses[-1].pose.position.x),
                                       (goal[1] - self.__path.poses[-1].pose.position.y))
            dist_start = self.__mean_sqare_dist_((start[0] - self.__path.poses[0].pose.position.x),
                                                (start[1] - self.__path.poses[0].pose.position.y))
        if not is_available or dist_goal > 0.5 or dist_start > 0.5:
            is_available = False
            print("path not valid!")
        return is_available

    def __generate_rand_pos_near_pos(self, path_pose, max_r, alpha):
        """
        Generating a random position2 close to another position1 within a radius of max_r.
        :param path_pose position1
        :param max_r radius in that position2 is generated
        :param alpha approx angle between position1 and position2
        :return [x, y] position2

        """

        pos_on_map = False
        r = max_r
        while not pos_on_map:
            if (random.random() > 0.5):
                alpha = alpha + random.random() * 45 * math.pi/180
            else:
                alpha = alpha - random.random() * 45 * math.pi/180

            x = r * math.cos(alpha) + path_pose.pose.position.x
            y = r * math.sin(alpha) + path_pose.pose.position.y
            pos_on_map = self.__is_pos_valid(x, y, self.__map)
            r -= 0.3

        return [x, y]

    def __map_callback(self, data):
        """
        Receiving map from map topic
        :param map data
        :return:
        """
        self.__map = data
        return

    def __path_callback(self, data):
        self.__path = data

    def __goal_status_callback(self, data):
        """
        Recovery method for stable learning:
        Checking goal status callback from global planner.
        If goal is not valid, new goal will be published.
        :param status_callback
        """
        if len(data.status_list) > 0:
            last_element = data.status_list[-1]
            if last_element.status == 4:
                if(self.__move_base_status_id != last_element.goal_id.id):
                    # self.set_random_static_task()
                    self.__move_base_status_id = last_element.goal_id.id

        return

    def __mean_sqare_dist_(self, x, y):
        """
        Computing mean square distance of x and y
        :param x, y
        :return: sqrt(x^2 + y^2)
        """
        return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

    def __yaw_to_quat(self, yaw):
        """
          Computing corresponding quaternion q to angle yaw [rad]
          :param yaw
          :return: q
          """
        q = Quaternion(axis=[0, 0, 1], angle=yaw)
        return q.elements
