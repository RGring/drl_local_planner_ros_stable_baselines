# What is this repository for?
* Setup to train a local planner with reinforcement learning approaches from [stable baselines](https://github.com/hill-a/stable-baselines) integrated ROS
* Training in a simulator fusion of [Flatland](https://github.com/avidbots/flatland) and [pedsim_ros](https://github.com/srl-freiburg/pedsim_ros)
* local planner has been trained on static and dynamic obstacles: [video](https://www.youtube.com/watch?v=nHvpO0hVnAg)
* Link to [IROS Paper](http://ras.papercept.net/images/temp/IROS/files/0122.pdf)
* Link to [Master Thesis](https://tams.informatik.uni-hamburg.de/publications/2019/MSc_Ronja_Gueldenring.pdf) for more in depth information.

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
    (please install missing packages)

4. Setup virtual environment to be able to use python3 with ros (consider also requirements.txt)
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

2. Execute self-trained ppo-agent
    * Copy your trained agent in your "path_to_models"
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
    source <path_to_venv>/venv_p3/bin/activate 
    roslaunch rl_agent run_ppo_agent.launch mode:="train"
    ```
    * Open fourth terminal: 
    ```
    roslaunch rl_bringup rviz.launch ns:="sim1"
    ```
    * Set 2D Navigation Goal in rviz

# Run pretrained Agents
Note: To be able to load the pretrained agents, you need to install numpy version 1.17.0.
```
<path_to_venv>/venv_p3/bin/pip install numpy==1.17
```

### Run agent trained on raw data, discrete action space, stack size 1
1. Copy the example_agents in your "path_to_models"
2. Open first terminal: 
    ```
    roscore
    ```
3. Open second terminal for visualization: 
    ```
    roslaunch rl_bringup rviz.launch ns:="sim1"
    ```
4. Open third terminal: 
    ```
    roslaunch rl_bringup setup.launch ns:="sim1" rl_params:="rl_params_scan"
    ```
5. Open fourth terminal:
    ```
    source <path_to_venv>/venv_p3/bin/activate 
    roslaunch rl_agent run_1_raw_disc.launch mode:="train"
    ```
### Run agent trained on raw data, discrete action space, stack size 3
1. Step 1 - 4 are the same like in the first example
2. Open fourth terminal:
    ```
    source <path_to_venv>/venv_p3/bin/activate 
    roslaunch rl_agent run_3_raw_disc.launch mode:="train"
    ```

### Run agent trained on raw data, continuous action space, stack size 1
1. Step 1 - 4 are the same like in the first example
2. Open fourth terminal:
    ```
    source <path_to_venv>/venv_p3/bin/activate 
    roslaunch rl_agent run_1_raw_cont.launch mode:="train"
    ```

### Run agent trained on image data, discrete action space, stack size 1
1. Step 1 - 3 are the same like in the first example
4. Open third terminal: 
    ```
    roslaunch rl_bringup setup.launch ns:="sim1" rl_params:="rl_params_img"
    ```
5. Open fourth terminal:
    ```
    source <path_to_venv>/venv_p3/bin/activate 
    roslaunch rl_agent run_1_img_disc.launch mode:="train"
    ```


    


# Training in Docker
I set up a docker image, that allows you to train a DRL-agent in parallel simulation environments. Furthermore, it simplifies the deployment on a server. Using docker you don't need to follow the steps in the Installation section.

0. Build the Docker image (This will unfortunately take about 15 minutes):
```
cd drl_local_planner_ros_stable_baselines/docker
```
```
docker build -t ros-drl_local_planner .
```
## Training from scratch
1. In start_scripts/training_params/ppo2_params, define the agents training parameters.

    | Parameter               | Desctiption |
    |-------------------------|--------------|
    | agent_name              |  Number of timestamps the agent will be trained.             |
    | total_timesteps         | Number of timestamps the agent will be trained. |
    | policy |  see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) |
    | gamma |  see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) |
    | n_steps |  see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) |
    | ent_coef |  see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) |
    | learning_rate |  see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) |
    | vf_coef |  see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) |
    | max_grad_norm |  see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) |
    | lam |  see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) |
    | nminibatches |  see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) |
    | noptepochs |  see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) |
    | cliprange |  see [PPO2 Doc](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html) |
    | robot_radius | The radius if the robot footprint |
    | rew_func | The reward functions that should be used. They can be found and defined in rl_agent/src/rl_agent/env_utils/reward_container.py. |
    | num_stacks | State representation includes the current observation and (num_stacks - 1) previous observation. |
    | stack_offset | The number of timestamps between each stacked observation. |
    | disc_action_space | 0, if continuous action space. 1, if discrete action space. |
    | normalize | 0, if input should not be normalized. 1, if input should be normalized. |
    | stage | stage number of your training. It is supposed to be 0, if you train for the first time. If it is > 0, it loads the agent of the "pretrained_model_path" and continues training. |
    | pretrained_model_name | If stage > 0 this agent will be loaded and training can be continued. |
    | task_mode | - "ped" for training on pedestrians only; "static" for training on static objects only; "ped_static" for training on both, static |



2. There are some predefined agents. As example I will use the ppo2_1_raw_data_disc_0 in the training session.

    ```
    docker run --rm -d \
        -v <folder_to_save_data>:/data \
        -v drl_local_planner_ros_stable_baselines/start_scripts/training_params:/usr/catkin_ws/src/drl_local_planner_ros_stable_baselines/start_scripts/training_params \
        -e AGENT_NAME=ppo2_1_raw_data_disc_0 \
        -e NUM_SIM_ENVS=4 \
        ros-drl_local_planner
    ```

3. If you want to display the training in Rviz, run the docker container in the hosts network. In order to use rviz, the relevant packages need to be compiled on your machine.
    ```
    docker run --rm -d \
        -v <folder_to_save_data>:/data \
        -v drl_local_planner_ros_stable_baselines/start_scripts/training_params:/usr/catkin_ws/src/drl_local_planner_ros_stable_baselines/start_scripts/training_params \
        -e AGENT_NAME=ppo2_1_raw_data_disc_0 \
        -e NUM_SIM_ENVS=4 \
        --net=host \
        ros-drl_local_planner
    ```
    Now you can display the different simulation environments:
    * Simulation 1:
        ```
        roslaunch rl_bringup rviz.launch ns:="sim1"
        ```
    * Simulation 2:
        ```
        roslaunch rl_bringup rviz.launch ns:="sim2"
        ```
    * etc. ...

## Train with pre-trained agents
### Run agent trained on raw data, discrete action space, stack size 1
    ```
    docker run --rm -d \
        -v drl_local_planner_ros_stable_baselines/example_agents:/data/agents \
        -v drl_local_planner_ros_stable_baselines/start_scripts/training_params:/usr/catkin_ws/src/drl_local_planner_ros_stable_baselines/start_scripts/training_params \
        -e AGENT_NAME=ppo2_1_raw_data_disc_0_pretrained \
        -e NUM_SIM_ENVS=4 \
        --net=host \
        ros-drl_local_planner
    ```
### Run agent trained on image data, discrete action space, stack size 1
    ```
    docker run --rm -d \
        -v drl_local_planner_ros_stable_baselines/example_agents:/data/agents \
        -v drl_local_planner_ros_stable_baselines/start_scripts/training_params:/usr/catkin_ws/src/drl_local_planner_ros_stable_baselines/start_scripts/training_params \
        -e AGENT_NAME=ppo2_1_img_disc_1_pretrained \
        -e NUM_SIM_ENVS=4 \
        --net=host \
        ros-drl_local_planner
    ```



    
    
