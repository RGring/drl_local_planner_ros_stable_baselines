'''
    @name:      analysis.py
    @brief:     Analysis of evaluation data
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''
import os
home = os.path.expanduser("~")
import rospkg
from rl_agent.evaluation.Analysis_eval import Analysis
import pickle
import configparser

def analyse(complexity, evaluation_file_path, reward_file_path, save_path):
    analysis = Analysis()
    results = analysis.load_results(evaluation_file_path)

    print("loaded data.")
    timesteps = analysis.get_timestep_list(results)
    success, time_exceeded, collision = analysis.get_scores(results)

    perc_success_drive = analysis.get_percentual_success_drive(results)

    path_ratio = analysis.get_path_length_ratio(results)

    speed = analysis.get_speed(results)

    if (complexity == "train"):
        [timesteps_tb, reward] = analysis.get_reward(reward_file_path)

    print("saving results...")

    # Collecting all results in a dict
    analysis_results = {"timesteps": timesteps,
                        "success": success,
                        "time_exceeded": time_exceeded,
                        "collision": collision,
                        "perc_success_drive": perc_success_drive,
                        "path_ratio": path_ratio,
                        "speed": speed}

    # Saving results
    if complexity == "train":
        analysis_results["timesteps_tb"] = timesteps_tb
        analysis_results["reward"] = reward

    with open('%s.pickle' % (save_path), 'wb') as handle:
        pickle.dump(analysis_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    rospack = rospkg.RosPack()
    rl_agent_path = rospack.get_path('rl_agent')
    config = configparser.ConfigParser()
    config.read('%s/config/path_config.ini' % rl_agent_path)
    path_to_eval_data_train = config['PATHES']['path_to_eval_data_train']
    path_to_eval_data_test = config['PATHES']['path_to_eval_data_test']

    ##########################################
    #### Training/Testin setup properties ####
    #### Please adjust.                   ####
    ##########################################
    complexity = "train"                # train, simple, average, complex or follow_path
    task_type = "static"                # static or ped
    no_episodes = 300
    # agent_names = ["ppo_195_10008246", "ppo_196_10002196", "ppo_197_10003358"]
    # agent_names = ["ppo2_48_9000600", "ppo2_47_9500400", "ppo2_46_10000200"]
    # agent_names = ["ppo2_43_10000200", "ppo2_44_9500400", "ppo2_45_10000200"]
    agent_names = ["ppo_195_10008246", "ppo_196_10002196", "ppo_197_10003358"]


    for agent_name in agent_names:
        if complexity == "train":
            evaluation_file_path = "%s/%s_training"%(path_to_eval_data_train, agent_name)
        elif complexity == "follow_path":
            evaluation_file_path = "%s/%s_following_path"%(path_to_eval_data_test, agent_name)
        else:
            evaluation_file_path = "%s/%s_%s_eval_set_average_%d" % (path_to_eval_data_test, agent_name, task_type, no_episodes)

        reward_file_path = "%s/run_%s_episode_reward" % (path_to_eval_data_train, agent_name)

        # Saving results
        if complexity == "train":
            save_path = "%s/%s_analysis.pickle" % (path_to_eval_data_train, agent_name)
        elif complexity == "follow_path":
            save_path = "%s/%s_analysis_follow_path" % (path_to_eval_data_test, agent_name)
        else:
            save_path = "%s/%s_%s_eval_set_%s_%s" % (
                path_to_eval_data_test, agent_name, task_type, complexity, no_episodes)

        analyse(complexity, evaluation_file_path, reward_file_path, save_path)

