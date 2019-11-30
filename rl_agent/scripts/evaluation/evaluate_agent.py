'''
    @name:      evaluate_agent.py
    @brief:     Evaluates an agent according to a test set.
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''
import os
home = os.path.expanduser("~")

import rospy
import rospkg
from multiprocessing import Process
from scripts.run_scripts.run_ppo import run_ppo
from rl_agent.env_utils.state_collector import StateCollector
from rl_agent.evaluation.Evaluation import Evaluation
import time
import configparser


def evaluate(ns, sc, evaluation_set_path, save_path):
    rospy.init_node("evaluate_node", anonymous=True)
    eval = Evaluation(sc, ns)
    time.sleep(2)
    eval.evaluate_set(evaluation_set_path, save_path)


if __name__ == '__main__':
    task_type = "ped"                       # static, ped
    complexity = "complex_map_3"            # simple, average, complex
    no_episodes = 700
    ns = "sim1"
    approach = "PPO2"                       # PPO1, PPO2
    policy = ["CnnPolicy", "CNN1DPolicy_multi_input", "CnnPolicy", "CnnPolicy"]
    disc_action_space = [False, True, True, True]
    agent_names = ["ppo2_109", "ppo2_99", "ppo2_107", "ppo2_108"]
    num_stacks = [1,1,1,3]
    stack_offset = 15

    rospack = rospkg.RosPack()
    rl_bringup_path = rospack.get_path('rl_bringup')
    config = configparser.ConfigParser()
    config.read('%s/config/path_config.ini' % rl_bringup_path)
    path_to_eval_sets = config['PATHES']['path_to_eval_sets']
    path_to_eval_data_test = config['PATHES']['path_to_eval_data_test']

    evaluation_set_name = "%s_eval_set_%s_%d"%(task_type, complexity, no_episodes)
    evaluation_set_path = "%s/%s"%(path_to_eval_sets, evaluation_set_name)
    mode = "exec"

    for i, agent_name in enumerate(agent_names):
        save_path = "%s/%s_%s" % (path_to_eval_data_test, agent_name, evaluation_set_name)
        sc = StateCollector(ns, "eval")

        p = Process(target=run_ppo, args=(config, sc, approach, agent_name , policy[i] , mode, task_type,
                                          stack_offset, num_stacks[i], True, False, disc_action_space[i], ns))
        p.start()

        print("Starting evaluation of agent %s with set %s"%(agent_name, evaluation_set_name))
        print("--------------------------------------------------------------------------------------")
        p_eval = Process(target=evaluate, args=(ns, sc, evaluation_set_path, save_path))
        p_eval.start()
        p_eval.join()
        p.terminate()


