#! /usr/bin/env python

'''
    @name:      train_ppo1.py
    @brief:     Starts a ppo1-training process. It is expected that move_base, simulation etc is started.(roslaunch rl_setup_bringup setup.launch)
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''
import os
home = os.path.expanduser("~")
import sys
import rospy
import rospkg
import configparser
from rl_agent.env_wrapper.ros_env_cont_img import RosEnvContImg
from rl_agent.env_wrapper.ros_env_cont_raw_data import RosEnvContRaw
from rl_agent.env_wrapper.ros_env_disc_raw_data import RosEnvDiscRaw
from rl_agent.env_wrapper.ros_env_cont_img_vel import RosEnvContImgVel
from rl_agent.env_wrapper.ros_env_disc_img_vel import RosEnvDiscImgVel
from rl_agent.env_wrapper.ros_env_cont_raw_scan_prep_wp import RosEnvContRawScanPrepWp
from rl_agent.env_wrapper.ros_env_disc_raw_scan_prep_wp import RosEnvDiscRawScanPrepWp
from rl_agent.env_wrapper.ros_env_disc_img import RosEnvDiscImg
from rl_agent.env_utils.state_collector import StateCollector
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from rl_agent.evaluation.Evaluation import Evaluation
from multiprocessing import Process
from rl_agent.common_custom_policies import *
from stable_baselines.common.policies import *
from stable_baselines.ppo1 import PPO1


def load_train_env(state_collector, ns,  robot_radius, rew_fnc, num_stacks, debug, task_mode, policy, disc_action_space, normalize):
    # Choosing environment wrapper according to the policy
    if policy == "CnnPolicy" or policy == "CnnLnLstmPolicy" or policy == "CnnLstmPolicy":
        if disc_action_space:
            env_temp = RosEnvDiscImg
        else:
            env_temp = RosEnvContImg
    elif policy == "CNN1DPolicy":
        if disc_action_space:
            env_temp = RosEnvDiscRawScanPrepWp
        else:
            env_temp = RosEnvContRawScanPrepWp
    elif policy == "CNN1DPolicy_multi_input":
        if disc_action_space:
            env_temp = RosEnvDiscRaw
        else:
            env_temp = RosEnvContRaw
    elif policy == "CnnPolicy_multi_input_vel" or policy == "CnnPolicy_multi_input_vel2":
        if disc_action_space:
            env_temp = RosEnvDiscImgVel
        else:
            env_temp = RosEnvContImgVel

    env_raw = DummyVecEnv([lambda: env_temp(ns, state_collector, robot_radius, rew_fnc, debug, "train", task_mode)])

    if normalize:
        env = VecNormalize(env_raw, training=True, norm_obs=True, norm_reward=False, clip_obs=100.0, clip_reward=10.0,
                           gamma=0.99, epsilon=1e-08)
    else:
        env = env_raw
    return [env_raw, env]


def train_agent_ppo(config, ns, agent_name, state_collector, total_timesteps, policy, gamma=0.99, timesteps_per_actorbatch=256,
                    clip_param=0.2, entcoeff=0.01, optim_epochs=4, optim_stepsize=0.001,
                    optim_batchsize=64, lam=0.95, adam_epsilon=1e-05, schedule='linear',
                    robot_radius = 0.46, rew_fnc=3, num_stacks=1, disc_action_space = False,
                    debug=False, normalize=False,
                    stage=0, pretrained_model_path="", task_mode="static"):

    # Loading simulation environment
    env_raw, env = load_train_env(state_collector,
                                  ns,
                                  robot_radius,
                                  rew_fnc,
                                  num_stacks,
                                  debug,
                                  task_mode,
                                  policy,
                                  disc_action_space,
                                  normalize)

    # Define pathes to store things
    path_to_tensorboard_log = config['PATHES']['path_to_tensorboard_log']
    path_to_checkpoint = config['PATHES']['path_to_checkpoint']
    path_to_final_models = config['PATHES']['path_to_final_models']

    #Setting up model
    if stage==0:
        model = PPO1(eval(policy), env, gamma=gamma, timesteps_per_actorbatch=timesteps_per_actorbatch,
                     clip_param=clip_param, entcoeff=entcoeff, optim_epochs=optim_epochs,
                     optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize, lam=lam,
                     adam_epsilon=adam_epsilon, schedule=schedule, verbose=1,
                     tensorboard_log='%s'%(path_to_tensorboard_log))
    else:
        # Pretrained model is loaded to continue training.
        model = PPO1.load("%s/%s" % (path_to_checkpoint, pretrained_model_path.strip()), env,
                          tensorboard_log='%s'%(path_to_tensorboard_log))


    # Document agent
    print("Starting PPO Training of agent: %s" %(agent_name))
    print("------------------------------------------------------")
    print("gamma \t\t\t\t %f" %model.gamma)
    print("timesteps_per_actorbatch \t %d" %model.timesteps_per_actorbatch)
    print("clip_param \t\t\t %f" %model.clip_param)
    print("entcoeff \t\t\t %f" %model.entcoeff)
    print("optim_epochs \t\t\t %d" %model.optim_epochs)
    print("optim_stepsize \t\t\t %f" %model.optim_stepsize)
    print("optim_batchsize \t\t %d" %model.optim_batchsize)
    print("lam \t\t\t\t %f" %model.lam)
    print("adam_epsilon \t\t\t %f" %model.adam_epsilon)
    print("schedule \t\t\t %s" %model.schedule)
    print("total_timesteps \t\t %d" %total_timesteps)
    print("Policy \t\t\t\t %s" % policy)
    print("Normalized state %d" %normalize)
    print("discrete action space %d"% disc_action_space)
    print("Debug: %d "% debug)
    print("\n")

    # Starting training
    if stage == 0:
        model.learn(total_timesteps=total_timesteps, log_interval=100, tb_log_name=agent_name, save_path="%s/%s"%(path_to_checkpoint, agent_name))
    else:
        model.learn(total_timesteps=total_timesteps, log_interval=100, tb_log_name="%s_stage_%d"%(agent_name, stage), save_path="%s/%s_stage_%d"%(path_to_checkpoint, agent_name, stage))

    # Saving final model
    model.save("%s/%s" % (path_to_final_models, "%s_stage_%d" % (agent_name, stage)))

def evaluate_during_training(state_collector, ns, save_path, robot_radius):
    rospy.init_node("evaluate_node", anonymous=True)
    eval = Evaluation(state_collector, ns, robot_radius=robot_radius)
    eval.evaluate_training(save_path)

if __name__ == '__main__':
    record_evaluation_data = False
    rospack = rospkg.RosPack()
    rl_agent_path = rospack.get_path('rl_agent')
    config = configparser.ConfigParser()
    config.read('%s/config/path_config.ini'%rl_agent_path)
    path_to_eval_data_train = config['PATHES']['path_to_eval_data_train']

    # for running via ./entrypoint_ppo.sh
    if (len(sys.argv) > 3):
        sc = StateCollector("sim1", "train")
        save_path = "%s/%s_training"%(path_to_eval_data_train, str(sys.argv[1]))
        if record_evaluation_data:
            p = Process(target=evaluate_during_training, args=(sc, "sim1", save_path, float(sys.argv[14])))
            p.start()

        train_agent_ppo(config, "sim1", str(sys.argv[1]), sc, int(sys.argv[2]), str(sys.argv[3]), gamma=float(sys.argv[4]),
                        timesteps_per_actorbatch=int(sys.argv[5]), clip_param=float(sys.argv[6]),
                        entcoeff=float(sys.argv[7]), optim_epochs=int(sys.argv[8]),
                        optim_stepsize=float(sys.argv[9]), optim_batchsize=int(sys.argv[10]),
                        lam=float(sys.argv[11]), adam_epsilon=float(sys.argv[12]),
                        schedule=str(sys.argv[13]), robot_radius = float(sys.argv[14]),
                        rew_fnc=float(sys.argv[15]), num_stacks=int(sys.argv[16]),
                        disc_action_space = bool(int(sys.argv[17])),
                        normalize=bool(int(sys.argv[18])), stage=int(sys.argv[19]),
                        pretrained_model_path=str(sys.argv[20]),
                        task_mode=str(sys.argv[21]))

    # for quick testing
    else:

        ns = "sim1"
        sc = StateCollector(ns, "train")
        agent_name = "ppo_foo"
        robot_radius = 0.56
        save_path = "%s/%s_training"%(path_to_eval_data_train, agent_name)
        if record_evaluation_data:
            p = Process(target=evaluate_during_training, args=(sc, ns, save_path, robot_radius))
            p.start()

        train_agent_ppo(config,
                    ns,
                    agent_name,
                    sc,
                    gamma=0.99,
                    entcoeff=0.003,
                    clip_param=0.2,
                    optim_epochs=4,
                    timesteps_per_actorbatch=4096,
                    optim_batchsize=2048,
                    total_timesteps=10000000,
                    policy="CnnPolicy",
                    debug=True,
                    rew_fnc = 1,
                    num_stacks=1,
                    disc_action_space=True,
                    robot_radius = robot_radius,
                    stage=0,
                    pretrained_model_path="ppo_162_4005210",
                    task_mode="ped_static")
