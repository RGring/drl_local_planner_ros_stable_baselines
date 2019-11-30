'''
    @name:      save_test_episodes.py
    @brief:     For saving n different episodes. Each episode needs to be rewiewed by the user and the
                following key press, trigger the followin action:
                [Y] Episode is added to the evaluation set
                [n] Episode is rejected
                [s] set is saved
                [e] quit program
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''

import os
home = os.path.expanduser("~")

import rospy
import rospkg
import pickle
from rl_agent.env_utils.task_generator import TaskGenerator
from rl_agent.env_utils.state_collector import StateCollector
import math
from operator import itemgetter
import time
import configparser



if __name__ == '__main__':
    rospy.init_node("episode_collector", anonymous=True)

    ##########################################
    #### Training/Testin setup properties ####
    #### Please adjust.                   ####
    ##########################################
    ns = "sim1"
    name = ""
    task_type = "ped"                    # static, ped
    complexity = "complex_map_1"                  # simple, average, complex

    rospack = rospkg.RosPack()
    rl_bringup_path = rospack.get_path('rl_bringup')
    config = configparser.ConfigParser()
    config.read('%s/config/path_config.ini' % rl_bringup_path)
    path_to_eval_sets = config['PATHES']['path_to_eval_sets']
    save_path = "%s/%s_eval_set_%s"%(path_to_eval_sets, task_type, complexity)
    state_collector = StateCollector(ns, "train")
    t_g = TaskGenerator(ns, state_collector, 0.56)

    if name != "":
        with open('%s/%s.pickle' % (path_to_eval_sets, name), 'rb') as handle:
            episodes = pickle.load(handle)
            count = len(episodes)
    else:
        episodes = []
        count = 0

    # Sleeping so that py-Publisher has time to setup!
    time.sleep(1)
    while True:
        print("Episode details:")

        if (task_type == "static"):
            d = t_g.set_random_static_task()
            print("\t -Start: (%d, %d)"%(d["start"][0], d["start"][1]))
            print("\t -Goal: (%d, %d)"%(d["goal"][0], d["goal"][1]))
            print("\t -Obstacles")
            for i in range(len(d["static_objects"]["x"])):
                print("\t \t *%s: (%f, %f)"% (d["static_objects"]["model_name"][i], d["static_objects"]["x"][i], d["static_objects"]["y"][i]))
            print("\n")
        elif (task_type == "ped"):
            d = t_g.set_random_ped_task()
            for i in range(10):
                t_g.take_sim_step()
                time.sleep(0.05)

            print("\t -Start: (%d, %d)" % (d["start"][0], d["start"][1]))
            print("\t -Goal: (%d, %d)" % (d["goal"][0], d["goal"][1]))
            print("\t -Peds")
            ped_count = 0
            for ped in d["peds"]:
                ped_poses = []
                for i in ped[2]:
                    ped_poses.append(i)
                print("\t\t*Ped%d"%ped_count)
                for i in ped_poses:
                    print("\t\t ", i)
                ped_count +=1
            print("\n")

        else:
            print("wrong task_type!")
            break

        path_length = 0
        if (len(d["path"].poses) <=2):
            print("Path length <= 2")
            continue
        old_pose = d["path"].poses[0].pose.position
        for pose in d["path"].poses[1:]:
            new_pose = pose.pose.position
            path_length += math.sqrt(math.pow((new_pose.x - old_pose.x), 2) + math.pow((new_pose.y - old_pose.y), 2))
            old_pose = new_pose
        d["path_length"] = path_length
        print("path_length: %f"%path_length)
        print("path_start: (%f, %f)"%(d["path"].poses[0].pose.position.x, d["path"].poses[0].pose.position.y))

        char = str(input("Episode valid? [Y][n]"))
        if (char == "n"):
            print("Episode rejected!")
        elif (char == "" or char == "y" or char == "Y"):
            episodes.append(d)
            print("Added episode %d"%count)
            count+=1
        elif (char == "s"):
            print("Saving episodes...")
            episodes = sorted(episodes, key=itemgetter('path_length'))
            with open('%s_%d.pickle'%(save_path, count), 'wb') as handle:
                pickle.dump(episodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif (char == "e"):
            break
