'''
    @name:      train_ppo1.py
    @brief:     Starts a ppo2-training process. It is expected that move_base, simulation etc is started.(roslaunch rl_setup_bringup setup.launch)
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
from rl_agent.env_wrapper.ros_env_disc_img import RosEnvDiscImg
from rl_agent.env_utils.state_collector import StateCollector
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv, VecFrameStack
from rl_agent.evaluation.Evaluation import Evaluation
from multiprocessing import Process
import random
from rl_agent.common_custom_policies import *
from stable_baselines.common.policies import *

from stable_baselines.ppo2 import PPO2


def load_train_env(num_envs, robot_radius, rew_fnc, num_stacks, stack_offset, debug, task_mode, policy, disc_action_space, normalize):
    # Choosing environment wrapper according to the policy
    if policy == "CnnPolicy" or policy == "CnnLnLstmPolicy" or policy == "CnnLstmPolicy":
        if disc_action_space:
            env_temp = RosEnvDiscImg
        else:
            env_temp = RosEnvContImg
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

    env = SubprocVecEnv([lambda k=k: env_temp("sim%d" % (k+1), StateCollector("sim%s"%(k+1), "train") , robot_radius, rew_fnc, debug, "train", task_mode) for k in range(num_envs)])

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
                     stage=0, pretrained_model_path="", task_mode="static"):

    # Setting seed
    seed = random.randint(0,1000)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    random.seed(seed)



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

    # Define pathes to store things
    path_to_tensorboard_log = config['PATHES']['path_to_tensorboard_log']
    path_to_checkpoint = config['PATHES']['path_to_checkpoint']
    path_to_final_models = config['PATHES']['path_to_final_models']


    if stage==0:
        model = PPO2(eval(policy), env, gamma=gamma,
                     n_steps=n_steps, ent_coef=ent_coef,
                     learning_rate=learning_rate, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                     lam=lam, nminibatches=nminibatches, noptepochs=noptepochs,
                     cliprange=cliprange, verbose=1,
                     tensorboard_log='%s' % (path_to_tensorboard_log))
    else:
        # Pretrained model is loaded to continue training.
        model = PPO2.load("%s/%s" % (path_to_checkpoint, pretrained_model_path.strip()), env,
                          tensorboard_log='%s'%(path_to_tensorboard_log))

    # Document agent
    rospy.loginfo("Starting PPO2 Training of agent: %s" %(agent_name))
    rospy.loginfo("------------------------------------------------------")
    rospy.loginfo("gamma \t\t\t\t %f" %model.gamma)
    rospy.loginfo("n_steps \t\t\t %d" %model.n_steps)
    rospy.loginfo("ent_coef \t\t\t %f" %model.ent_coef)
    rospy.loginfo("learning_rate \t\t\t %f" %learning_rate)
    rospy.loginfo("vf_coef \t\t\t %f" %model.vf_coef)
    rospy.loginfo("max_grad_norm \t\t\t %f" %model.max_grad_norm)
    rospy.loginfo("lam \t\t\t\t %f" %model.lam)
    rospy.loginfo("nminibatches \t\t\t %d" %model.nminibatches)
    rospy.loginfo("noptepochs \t\t\t %d" %model.noptepochs)
    rospy.loginfo("cliprange \t\t\t %f" %cliprange)
    rospy.loginfo("total_timesteps \t\t %d" %total_timesteps)
    rospy.loginfo("Policy \t\t\t\t %s" %policy)
    rospy.loginfo("reward_fnc \t\t\t %d" %rew_fnc)
    rospy.loginfo("Normalized state: %d" % normalize)
    rospy.loginfo("discrete action space %d" % disc_action_space)
    rospy.loginfo("Number of stacks: %d, stack offset: %d" % (num_stacks, stack_offset))
    rospy.loginfo("\n")

    # Starting training
    if stage == 0:
        model.learn(total_timesteps=total_timesteps, log_interval=100, tb_log_name=agent_name, save_path="%s/%s"%(path_to_checkpoint, agent_name))
    else:
        model.learn(total_timesteps=total_timesteps, log_interval=100, tb_log_name="%s_stage_%d"%(agent_name, stage), save_path="%s/%s_stage_%d"%(path_to_checkpoint, agent_name, stage))

    # Saving final model
    model.save("%s/%s/%s" % (path_to_final_models, "%s_stage_%d" % (agent_name, stage)))

def evaluate_during_training(ns, save_path, robot_radius):
    rospy.init_node("evaluate_node", anonymous=True)
    eval = Evaluation(StateCollector(ns, "train"), ns, robot_radius=robot_radius)
    eval.evaluate_training(save_path)

if __name__ == '__main__':
    record_evaluation_data = False

    rospack = rospkg.RosPack()
    rl_agent_path = rospack.get_path('rl_agent')
    config = configparser.ConfigParser()
    config.read('%s/config/path_config.ini'%rl_agent_path)
    path_to_eval_data_train = config['PATHES']['path_to_eval_data_train']

    # for running via ./entrypoint_ppo2.sh
    if (len(sys.argv) > 1):
        save_path = "%s/%s_training"%(path_to_eval_data_train, str(sys.argv[1]))
        if record_evaluation_data:
            for i in range(int(sys.argv[22])):
                p = Process(target=evaluate_during_training, args=("sim%d" % (i + 1), save_path, float(sys.argv[14])))
                p.start()
        train_agent_ppo2(config, str(sys.argv[1]), int(sys.argv[2]), str(sys.argv[3]), gamma=float(sys.argv[4]),
                         n_steps=int(sys.argv[5]), ent_coef=float(sys.argv[6]),
                         learning_rate=float(sys.argv[7]), vf_coef=float(sys.argv[8]),
                         max_grad_norm=float(sys.argv[9]), lam=float(sys.argv[10]),
                         nminibatches=int(sys.argv[11]), noptepochs=int(sys.argv[12]),
                         cliprange=float(sys.argv[13]), robot_radius=float(sys.argv[14]),
                         rew_fnc=float(sys.argv[15]), num_stacks=int(sys.argv[16]),
                         disc_action_space=bool(int(sys.argv[17])), normalize=bool(int(sys.argv[18])),
                         stage=int(sys.argv[19]),
                         pretrained_model_path = str(sys.argv[20]),
                         task_mode=str(sys.argv[21]), num_envs=int(sys.argv[22]))

    # for quick testing
    else:
        num_envs = 1
        agent_name = "ppo2_foo"
        robot_radius = 0.56
        save_path = "%s/%s_training"%(path_to_eval_data_train, agent_name)

        if record_evaluation_data:
            for i in range(num_envs):
                p = Process(target=evaluate_during_training, args=("sim%d"%(i+1), save_path, robot_radius))
                p.start()

        train_agent_ppo2(config,
                         agent_name,
                         gamma=0.99,
                         n_steps=128,
                         ent_coef=0.005,
                         learning_rate=0.00025,
                         cliprange=0.2,
                         total_timesteps=500000,
                         policy="CnnPolicy",
                         num_envs=num_envs,
                         nminibatches=1,
                         noptepochs=1,
                         debug=True,
                         rew_fnc = 2.1,
                         num_stacks=1,
                         disc_action_space=True,
                         robot_radius = robot_radius,
                         stage=0,
                         pretrained_model_path="ppo_162_4005210",
                         task_mode="ped")
