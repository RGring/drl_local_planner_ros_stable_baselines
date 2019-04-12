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
    rospack = rospkg.RosPack()
    rl_agent_path = rospack.get_path('rl_agent')
    config = configparser.ConfigParser()
    config.read('%s/config/path_config.ini' % rl_agent_path)
    path_to_eval_sets = config['PATHES']['path_to_eval_sets']
    path_to_eval_data_test = config['PATHES']['path_to_eval_data_test']

    ##########################################
    #### Training/Testin setup properties ####
    #### Please adjust.                   ####
    ##########################################
    task_type = "static"            # static, ped
    complexity = "average"          # simple, average, complex
    no_episodes = 300
    ns = "sim1"
    approach = "PPO1"               # PPO1, PPO2
    policy = "CnnPolicy"            # CnnPolicy, CNN1DPolicy, CNN1DPolicy_multi_input, CnnPolicy_multi_input_vel
    disc_action_space = True
    # agent_names = ["ppo_195_10008246", "ppo_196_10002196", "ppo_197_10003358"]
    agent_names = ["ppo2_47_9500400", "ppo2_48_9000600", "ppo2_46_10000200"]


    evaluation_set_name = "%s_eval_set_%s_%d"%(task_type, complexity, no_episodes)
    evaluation_set_path = "%s/%s"%(path_to_eval_sets, evaluation_set_name)
    mode = "exec"

    for agent_name in agent_names:
        save_path = "%s/%s_%s" % (path_to_eval_data_test, agent_name, evaluation_set_name)
        sc = StateCollector(ns, "eval")

        p = Process(target=run_ppo, args=(sc, approach, agent_name , policy , mode, task_type, 1, True, False, disc_action_space, ns))
        p.start()

        print("Starting evaluation of agent %s with set %s"%(agent_name, evaluation_set_name))
        print("--------------------------------------------------------------------------------------")
        p_eval = Process(target=evaluate, args=(ns, sc, evaluation_set_path, save_path))
        p_eval.start()
        p_eval.join()

        p.terminate()


