'''
    @name:      train_ppo.py
    @brief:     Starts a ppo2-training process. It is expected that move_base, simulation etc is started.(roslaunch rl_setup_bringup setup.launch)
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''
import os
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
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv, VecFrameStack
from rl_agent.evaluation.Evaluation import Evaluation
from multiprocessing import Process
import random
from rl_agent.common_custom_policies import *
from stable_baselines.common.policies import *

from stable_baselines.ppo2 import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np

best_mean_reward, n_callback = -np.inf, 0
agent_name = ""
path_to_models = ""


def train_callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_callback, best_mean_reward, agent_name, path_to_models
  # Print stats every 1000 calls
  if (n_callback + 1) % 10 == 0:

      # Evaluate policy performance
      x, y = ts2xy(load_results('%s/%s/'%(path_to_models, agent_name)), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(path_to_models + '/%s/%s.pkl' % (agent_name, agent_name))
  n_callback += 1
  return True

def load_train_env(num_envs, robot_radius, rew_fnc, num_stacks, stack_offset, debug, task_mode, policy, disc_action_space, normalize):
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

    env = SubprocVecEnv([lambda k=k: Monitor(env_temp("sim%d" % (k+1), StateCollector("sim%s"%(k+1), "train") , stack_offset, num_stacks, robot_radius, rew_fnc, debug, "train", task_mode), '%s/%s/sim_%d'%(path_to_models, agent_name, k+1), allow_early_resets=True) for k in range(num_envs)])

    # Normalizing?
    if normalize:
        env = VecNormalize(env, training=True, norm_obs=True, norm_reward=False, clip_obs=100.0, clip_reward=10.0,
                           gamma=0.99, epsilon=1e-08)
    else:
        env = env

    # Stack of data?
    if num_stacks > 1:
        env = VecFrameStack(env, n_stack=num_stacks, n_offset=stack_offset)

    return env

def train_agent_ppo2(config, agent_name, total_timesteps, policy,
                     gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=0.00025,
                     vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4,
                     cliprange=0.2, num_envs=1, robot_radius = 0.46, rew_fnc=3, num_stacks=1, stack_offset=15, disc_action_space = False,
                     debug=False, normalize=False,
                     stage=0, pretrained_model_name="", task_mode="static"):

    # Setting seed
    seed = random.randint(0,1000)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    random.seed(seed)

    # Define pathes to store things
    path_to_tensorboard_log = config['PATHES']['path_to_tensorboard_log']
    global path_to_models
    path_to_models = config['PATHES']['path_to_models']

    agent_dir='%s/%s'%(path_to_models, agent_name)
    if not os.path.exists(agent_dir):
        os.makedirs(agent_dir)



    # Loading simulation environment
    env = load_train_env(num_envs,
                                  robot_radius,
                                  rew_fnc,
                                  num_stacks,
                                  stack_offset,
                                  debug,
                                  task_mode,
                                  policy,
                                  disc_action_space,
                                  normalize)



    if stage==0:
        model = PPO2(eval(policy), env, gamma=gamma,
                     n_steps=n_steps, ent_coef=ent_coef,
                     learning_rate=learning_rate, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                     lam=lam, nminibatches=nminibatches, noptepochs=noptepochs,
                     cliprange=cliprange, verbose=1,
                     tensorboard_log='%s' % (path_to_tensorboard_log))
    else:
        # Pretrained model is loaded to continue training.
        model = PPO2.load("%s/%s/%s.pkl" % (path_to_models, pretrained_model_name, pretrained_model_name), env,
                          tensorboard_log='%s'%(path_to_tensorboard_log))

    # Document agent
    print("Starting PPO2 Training of agent: %s" %(agent_name))
    print("------------------------------------------------------")
    print("gamma \t\t\t\t %f" %model.gamma)
    print("n_steps \t\t\t %d" %model.n_steps)
    print("ent_coef \t\t\t %f" %model.ent_coef)
    print("learning_rate \t\t\t %f" %learning_rate)
    print("vf_coef \t\t\t %f" %model.vf_coef)
    print("max_grad_norm \t\t\t %f" %model.max_grad_norm)
    print("lam \t\t\t\t %f" %model.lam)
    print("nminibatches \t\t\t %d" %model.nminibatches)
    print("noptepochs \t\t\t %d" %model.noptepochs)
    print("cliprange \t\t\t %f" %cliprange)
    print("total_timesteps \t\t %d" %total_timesteps)
    print("Policy \t\t\t\t %s" %policy)
    print("reward_fnc \t\t\t %d" %rew_fnc)
    print("Normalized state: %d" % normalize)
    print("discrete action space %d" % disc_action_space)
    print("Number of stacks: %d, stack offset: %d" % (num_stacks, stack_offset))
    print("\n")

    # Starting training
    reset_num_timesteps = False
    if stage==0:
        reset_num_timesteps = True

    model.learn(total_timesteps=total_timesteps, log_interval=100, callback=train_callback, tb_log_name=agent_name, reset_num_timesteps=reset_num_timesteps)

    # Saving final model
    model.save("%s/%s/%s" % (path_to_models, agent_name, "%s_stage_%d" % (agent_name, stage)))
    print("Training finished.")
    env.close()

def evaluate_during_training(ns, save_path, robot_radius):
    rospy.init_node("evaluate_node", anonymous=True)
    eval = Evaluation(StateCollector(ns, "train"), ns, robot_radius=robot_radius)
    eval.evaluate_training(save_path)

if __name__ == '__main__':
    record_evaluation_data = False

    rospack = rospkg.RosPack()
    rl_bringup_path = rospack.get_path('rl_bringup')
    config = configparser.ConfigParser()
    config.read('%s/config/path_config.ini'%rl_bringup_path)
    path_to_eval_data_train = config['PATHES']['path_to_eval_data_train']

     # for running via ./entrypoint_ppo2.sh
    if (len(sys.argv) > 1):
        agent_name = str(sys.argv[1])
        stage = int(sys.argv[20])

        record_processes = []
        if record_evaluation_data:
            save_path = "%s/%s_training" % (path_to_eval_data_train, str(sys.argv[1]))
            for i in range(int(sys.argv[23])):
                p = Process(target=evaluate_during_training, args=("sim%d" % (i + 1), save_path, float(sys.argv[14])))
                p.start()
                record_processes.append(p)


        train_agent_ppo2(config, agent_name, int(sys.argv[2]), str(sys.argv[3]), gamma=float(sys.argv[4]),
                         n_steps=int(sys.argv[5]), ent_coef=float(sys.argv[6]),
                         learning_rate=float(sys.argv[7]), vf_coef=float(sys.argv[8]),
                         max_grad_norm=float(sys.argv[9]), lam=float(sys.argv[10]),
                         nminibatches=int(sys.argv[11]), noptepochs=int(sys.argv[12]),
                         cliprange=float(sys.argv[13]), robot_radius=float(sys.argv[14]),
                         rew_fnc=float(sys.argv[15]), num_stacks=int(sys.argv[16]),
                         stack_offset=int(sys.argv[17]),
                         disc_action_space=bool(int(sys.argv[18])), normalize=bool(int(sys.argv[19])),
                         stage=stage,
                         pretrained_model_name = str(sys.argv[21]),
                         task_mode=str(sys.argv[22]), num_envs=int(sys.argv[23]))

        for p in record_processes:
            p.terminate()

    # for quick testing
    else:

        num_envs = 1
        stage = 1
        agent_name = "ppo2_test_continue_train"
        robot_radius = 0.5

        record_processes = []
        if record_evaluation_data:
            save_path = "%s/%s_training" % (path_to_eval_data_train, agent_name)
            for i in range(num_envs):
                p = Process(target=evaluate_during_training, args=("sim%d"%(i+1), save_path, robot_radius))
                p.start()
                record_processes.append(p)

        train_agent_ppo2(config,
                         agent_name,
                         gamma=0.99,
                         n_steps=128,
                         ent_coef=0.005,
                         learning_rate=0.00025,
                         cliprange=0.2,
                         total_timesteps=10000000,
                         policy="CNN1DPolicy_multi_input",
                         num_envs=num_envs,
                         nminibatches=1,
                         noptepochs=1,
                         debug=True,
                         rew_fnc = 19,
                         num_stacks= 3,
                         stack_offset=5,
                         disc_action_space=False,
                         robot_radius = robot_radius,
                         stage=stage,
                         pretrained_model_name="ppo2_test_continue_train",
                         task_mode="ped")

        for p in record_processes:
            p.terminate()
