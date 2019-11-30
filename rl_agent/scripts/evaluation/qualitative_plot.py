'''
    @name:      qualitative_plot.py
    @brief:     Generates a situation in rviz, that shows the agents qualitative behaviour.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''
import os
home = os.path.expanduser("~")

import rospy
import rospkg
from rl_agent.evaluation.Analysis_eval import Analysis
from rl_agent.evaluation.Evaluation import Evaluation
from rl_agent.env_utils.state_collector import StateCollector
import time
import configparser


if __name__ == '__main__':
    rospy.init_node("qualitative_image_generation_node", anonymous=True)

    ##########################################
    #### Training/Testin setup properties ####
    #### Please adjust.                   ####
    ##########################################
    ns = "sim1"
    complexity = "average"              # simple, average, complex
    task_type = "static"                # static, ped
    no_episodes = 3
    agent_name_1 = "ppo2_foo_512"
    agent_name_2 = ""

    rospack = rospkg.RosPack()
    rl_bringup_path = rospack.get_path('rl_bringup')
    config = configparser.ConfigParser()
    config.read('%s/config/path_config.ini' % rl_bringup_path)
    path_to_eval_sets = config['PATHES']['path_to_eval_sets']
    path_to_eval_data_test = config['PATHES']['path_to_eval_data_test']
    path_to_agent_1_results = "%s/%s_static_eval_set_%s_%d" % (path_to_eval_data_test, agent_name_1, complexity, no_episodes)
    path_to_agent_2_results = "%s/%s_static_eval_set_%s_%d" % (path_to_eval_data_test, agent_name_2, complexity, no_episodes)
    path_to_evaluation_set = "%s/%s_eval_set_%s_%d"%(path_to_eval_sets, task_type, complexity, no_episodes)
    print("Loading sets...")

    analysis = Analysis()
    results_agent_2 = []        # Empty, if no second results are needed.
    results_agent_1 = analysis.load_results(path_to_agent_1_results)
    if(task_type == "static"):
        if agent_name_2 != "":
            results_agent_2 = analysis.load_results(path_to_agent_1_results)

    eval = Evaluation(StateCollector(ns, "eval"), ns)
    time.sleep(2)
    print("Printing set.")
    len_eval_set = eval.load_evaluation_set(path_to_evaluation_set)

    for i in range(0, len_eval_set):
        episode_pos = 10
        while episode_pos >= 0:
            if (task_type == "static"):
                eval.generate_qualitative_static_image_rviz(results_agent_1, results_agent_2, i, episode_pos)
            elif (task_type == "ped"):
                eval.generate_qualitative_ped_image_rviz(results_agent_1, i, episode_pos)
            episode_pos =  int(input("Pos of episode %d ? (-1 for plotting next episode)" % i) or "50")


