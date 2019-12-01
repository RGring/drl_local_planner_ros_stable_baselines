FROM ros:kinetic-robot-xenial

RUN apt-get update && apt-get install -y \
  libqt4-dev \
  libopencv-dev \
  liblua5.2-dev \
  virtualenv \
  screen \
  python3-dev \
  git \
  ros-kinetic-tf2-geometry-msgs \
  ros-kinetic-navigation \
  ros-kinetic-rviz 

# Building catkin_ws
RUN mkdir -p /usr/catkin_ws/src
WORKDIR /usr/catkin_ws/src
RUN git clone https://github.com/RGring/drl_local_planner_ros_stable_baselines
RUN cp drl_local_planner_ros_stable_baselines/.rosinstall .
RUN rosws update
RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash; cd /usr/catkin_ws; catkin_make'

#Creating virtualenv for python3
RUN virtualenv /venv_p3 --python=python3
RUN /venv_p3/bin/pip install \
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
RUN /venv_p3/bin/pip install -e /usr/catkin_ws/src/drl_local_planner_forks/stable-baselines


RUN mv /usr/catkin_ws/src/drl_local_planner_ros_stable_baselines/rl_bringup/config/path_config_docker.ini \
/usr/catkin_ws/src/drl_local_planner_ros_stable_baselines/rl_bringup/config/path_config.ini

WORKDIR /usr/catkin_ws/src/drl_local_planner_ros_stable_baselines/start_scripts/

ENTRYPOINT ./entrypoint_ppo2.sh "$AGENT_NAME" "$NUM_SIM_ENVS"
