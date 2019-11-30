# What is this repository for?
* Setup to train a local planner with reinforcement learning approaches from [stable baselines](https://github.com/hill-a/stable-baselines) integrated ROS
* Training in a simulator fusion of [Flatland](https://github.com/avidbots/flatland) and [pedsim_ros](https://github.com/srl-freiburg/pedsim_ros)
* local planner has been trained on static and dynamic obstacles: [video](https://www.youtube.com/watch?v=laGrLaMaeT4)

# Installation (Else: Docker below)

1. Standart ROS setup (Code has been tested with ROS-kinetic on Ubuntu 16.04)

2. Install additional packages
    ```
    apt-get update && apt-get install -y \
    libqt4-dev \
    libopencv-dev \
    liblua5.2-dev \
    virtualenv \
    screen \
    python3-dev \
    ros-kinetic-tf2-geometry-msgs \
    ros-kinetic-navigation \
    ros-kinetic-rviz 
    ```

3. Setup repository: 
    * Clone this repository in your src-folder of your catkin workspace
    ```
    cd <path_to_catkin_ws>/src/drl_local_planner_ros_stable_baselines
    cp .rosinstall ../
    cd ..
    rosws update
    cd <path_to_catkin_ws>
    catkin_make -DCMAKE_BUILD_TYPE=Release
    ```
    (please install required ros packages)

4. Setup virtual environment to be able to use python3 with ros
    ```
    virtualenv <path_to_venv>/venv_p3 --python=python3
    source <path_to_venv>/venv_p3/bin/activate
    <path_to_venv>/venv_p3/bin/pip install \
        pyyaml \
        rospkg \
        catkin_pkg \
        exception \
        numpy \
        tensorflow=="1.13.1" \
        gym \
        pyquaternion \ 
        mpi4py \
        matplotlib
    cd <path_to_catkin_ws>/src/drl_local_planner_forks/stable_baselines/
    <path_to_venv>/venv_p3/bin/pip install -e path_to_catkin_ws>/src/drl_local_planner_forks/stable-baselines/
    ```
5. Set system-relevant variables 
    * Modify all relevant pathes rl_bringup/config/path_config.ini


# Example usage

1. Train agent
    * Open first terminal (roscore): 
    ```
    roscore
    ```
    * Open second terminal (simulationI:
    ```
    roslaunch rl_bringup setup.launch ns:="sim1" rl_params:="rl_params_scan"
    ```
    * Open third terminal (DRL-agent):
     ```
    source <path_to_venv>/bin/activate 
    python rl_agent/scripts/train_scripts/train_ppo.py
    ```
    * Open fourth terminal (Visualization):
     ```
    roslaunch rl_bringup rviz.launch ns:="sim1"
    ```

2. Execute trained ppo-agent
    * Copy the example_agents in your "path_to_models"
    * Open first terminal: 
    ```
    roscore
    ```
    * Open second terminal: 
    ```
    roslaunch rl_bringup setup.launch ns:="sim1" rl_params:="rl_params_scan"
    ```
    * Open third terminal:
    ```
    source <path_to_venv>/bin/activate 
    roslaunch rl_agent run_ppo_agent.launch mode:="train"
    ```
    * Open fourth terminal: 
    ```
    roslaunch rl_bringup rviz.launch ns:="sim1"
    ```
    * Set 2D Navigation Goal in rviz


# Docker
I set up a docker image, that allows you to train a DRL-agent in parallel simulation environments. Using docker you don't need to follow the steps in the Installation section.

0. Build the Docker image (This will unfortunately take about 15 minutes):
'''
docker build
'''

1. In start_scripts/training_params/ppo2_params, define the agents training parameters.
|       | computational time[sec] | actions used |
|-------|-------------------------|--------------|
| mod1  |                         |              |
| mod2  | 41.9453                 | 1777         |
| final | 6.8229                  | 1826         |
    * agent_name - Name of the agent
    * total_timesteps - Number of timestamps the agent will be trained.
    * policy - see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * gamma - [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * n_steps - [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * ent_coef - [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * learning_rate - [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * vf_coef - [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * max_grad_norm - [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * lam - [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * nminibatches - [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * noptepochs - [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * cliprange - [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
    * robot_radius  
    >       The radius of the robots footprint.
    * rew_fnc  
    >       Number of reward function, that should be used. Can be found in rl_agent/src/rl_agent/env_utils/reward_container.py.
    * num_stacks  
    >       State representation includes the current observation and (num_stacks - 1) previous observation.
    * stack_offset  
    >       The number of timestamps between each stacked observation.
    * disc_action_space  
    >       0, if continuous action space. 1, if discrete action space
    * normalize  
    >       0, if input should not be normalized. 1, if input should be normalized.
    * stage  
    >       stage number of your training. It is supposed to be 0, if you train for the first time. If it is > 0, it loads the agent of the "pretrained_model_path" and continues training.
    * pretrained_model_path  
    >       If stage > 0 this agent will be loaded and training can be continued.
    * task_mode 
    >       - "ped" for training on pedestrians only  
    >       - "static" for training on static objects only  
    >       - "ped_static" for training on both, static objects and pedestrians.

2. There are some predefined agents. I will start the docker with the ppo2_1_raw_data_disc_0-training-session.
'''
docker run -rm -it -v <folder_to_save_data>:/data \
    -e AGENT_NAME=ppo2_1_raw_data_disc_0 \
    ros-drl_local_planner
'''



    
    
