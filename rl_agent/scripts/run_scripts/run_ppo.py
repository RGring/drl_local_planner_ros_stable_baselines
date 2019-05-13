#! /usr/bin/env python
'''
    @name:      run_ppo.py
    @brief:     Trained agent is loaded and executed.
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
import time
import numpy as np
from rl_agent.env_wrapper.ros_env_cont_img import RosEnvContImg
from rl_agent.env_wrapper.ros_env_cont_raw_data import RosEnvContRaw
from rl_agent.env_wrapper.ros_env_disc_raw_data import RosEnvDiscRaw
from rl_agent.env_wrapper.ros_env_cont_raw_scan_prep_wp import RosEnvContRawScanPrepWp
from rl_agent.env_wrapper.ros_env_disc_raw_scan_prep_wp import RosEnvDiscRawScanPrepWp
from rl_agent.env_wrapper.ros_env_cont_img_vel import RosEnvContImgVel
from rl_agent.env_wrapper.ros_env_disc_img_vel import RosEnvDiscImgVel
from rl_agent.env_wrapper.ros_env_disc_img import RosEnvDiscImg
from rl_agent.env_utils.state_collector import StateCollector
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines.ppo2.ppo2 import PPO2
from stable_baselines.ppo1.pposgd_simple import PPO1

def load_train_env(ns, state_collector, robot_radius, rew_fnc, num_stacks,
                   stack_offset, debug, task_mode, rl_mode, policy, disc_action_space, normalize):
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

    env_raw = DummyVecEnv([lambda: env_temp(ns, state_collector, robot_radius, rew_fnc, debug, rl_mode, task_mode)])

    if normalize:
        env = VecNormalize(env_raw, training=True, norm_obs=True, norm_reward=False, clip_obs=100.0, clip_reward=10.0,
                           gamma=0.99, epsilon=1e-08)
    else:
        env = env_raw

    # Stack of data?
    if num_stacks > 1:
        env = VecFrameStack(env, n_stack=num_stacks, n_offset=stack_offset)

    return env


def run_ppo(config, state_collector, approach = "PPO1", agent_name ="ppo_99_8507750", policy ="CnnPolicy", mode="train", task_mode="static",
             stack_offset=15, debug=True, normalize = True, disc_action_space = False, ns=""):

    if approach != "PPO1" and approach != "PPO2":
        rospy.logerr("Wrong approach: %s.\n Only two different kind of approaches are excepted: PPO1, PPO2."%approach)
        return

    path_to_checkpoint = config['PATHES']['path_to_checkpoint']

    # Loading agent
    model = eval(approach).load("%s/%s" % (path_to_checkpoint, agent_name))

    print("Loaded %s" % agent_name)
    print("--------------------------------------------------")
    print("Normalize: ", normalize)
    print("Policy: %s" % policy)
    print("Discrete action space: ", disc_action_space)
    print("Observation space size: ", model.observation_space.shape)
    print("Debug: ", debug)
    print("Number of stacks: %d, stack offset: %d" % ( model.observation_space.shape[2], stack_offset))
    print("\n")


    #Loading environment
    env = load_train_env(ns, state_collector, 0.46, 1, model.observation_space.shape[2], stack_offset, True, task_mode, mode, policy, disc_action_space, normalize)

    # Resetting environment
    if mode == "train" or mode == "eval":
        obs = env.reset()
    if mode == "exec" or mode == "exec_rw":
        if disc_action_space:
            obs, rewards, dones, info = env.step([5])
        else:
            obs, rewards, dones, info = env.step([[0.0, 0.0]])

    if debug:
        #Cummulative reward.
        cum_reward = 0
    while True:
        #Determining action vor given observation
        action, _states = model.predict(obs)

        # Clipping actions
        if not disc_action_space:
            action = np.maximum(np.minimum(model.action_space.high, action), model.action_space.low)

        #Executing action in environment
        obs, rewards, dones, info = env.step(action)

        if debug:
            cum_reward += rewards

            # Episode over?
            if dones:
                print("Episode finished with reward of %f."% cum_reward)
                cum_reward = 0

        time.sleep(0.0001)
        if rospy.is_shutdown():
            print('shutdown')
            break


if __name__ == '__main__':

    rospack = rospkg.RosPack()
    rl_agent_path = rospack.get_path('rl_agent')
    config = configparser.ConfigParser()
    config.read('%s/config/path_config.ini'%rl_agent_path)

    # for running from terminal (e.g. launch-file)
    if (len(sys.argv) > 1):
        ns = "sim1"
        sc = StateCollector(ns, str(sys.argv[3]))
        run_ppo(config, sc,
                 ns=ns,
                 agent_name=str(sys.argv[1]),
                 policy=str(sys.argv[2]),
                 mode=str(sys.argv[3]),
                 debug=bool(int(sys.argv[4])),
                 normalize=bool(int(sys.argv[5])),
                 disc_action_space=bool(int(sys.argv[6])),
                 approach=str(sys.argv[7]),
                 task_mode=str(sys.argv[8]))

# for quick testing
    else:
        mode = "train"
        ns = "sim1"
        policy = "CnnPolicy_multi_input_vel2"
        approach = "PPO2"                       # Either PPO1 or PPO2
        agent_name = "ppo2_35_8001000"
        sc = StateCollector(ns, mode)

        run_ppo(config, sc,
                 agent_name= agent_name,
                 policy = policy,
                 mode = mode,
                 debug = True,
                 normalize=False,
                 disc_action_space=True,
                 task_mode="ped",
                 stack_offset = 15,
                 approach=approach,
                 ns=ns)

