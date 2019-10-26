'''
    @name:      state_collector.py
    @brief:     This class provides custom policies used in stable-baselines library
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''

from stable_baselines.common.policies import *
import stable_baselines.common.policies as common
import rospy
import numpy as np

NS="sim1"


######################################
###### CNN1DPolicy_multi_input #######
######################################

def laser_cnn_multi_input(state, **kwargs):
    """
    1D Conv Network

    :param state: (TensorFlow Tensor) state input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    # scan = tf.squeeze(state[:, : , 0:kwargs['laser_scan_len'] , :], axis=1)
    scan = tf.squeeze(state[:, : , 0:kwargs['laser_scan_len'] , :], axis=1)
    wps = tf.squeeze(state[:, :, kwargs['laser_scan_len']:, -1], axis=1)
    # goal = tf.math.multiply(goal, 6)

    kwargs_conv = {}
    activ = tf.nn.relu
    layer_1 = activ(conv1d(scan, 'c1d_1', n_filters=32, filter_size=5, stride=2, init_scale=np.sqrt(2), **kwargs_conv))
    layer_2 = activ(conv1d(layer_1, 'c1d_2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs_conv))
    layer_2 = conv_to_fc(layer_2)
    layer_3 = activ(linear(layer_2, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    temp = tf.concat([layer_3, wps], 1)
    layer_4 = activ(linear(temp, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))
    return layer_4



class CNN1DPolicy_multi_input(common.FeedForwardPolicy):
    """
    This class provides a 1D convolutional network for the Raw Data Representation
    """
    def __init__(self, *args, **kwargs):
        kwargs["laser_scan_len"] = rospy.get_param("%s/rl_agent/scan_size"%NS, 360)
        super(CNN1DPolicy_multi_input, self).__init__(*args, **kwargs, cnn_extractor=laser_cnn_multi_input, feature_extraction="cnn")



##########################
###### CNN1DPolicy #######
##########################
def laser_cnn_stack(state, **kwargs):
    """
    1D Conv Network

    :param state: (TensorFlow Tensor) Scan input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """

    scan = state[:, : , 0 , :]
    wps = tf.expand_dims(state[:, :, 1, -1], -1)
    scan_con_wps = tf.concat([scan, wps], -1)
    # goal = tf.math.multiply(goal, 6)

    kwargs_conv = {}
    activ = tf.nn.relu
    layer_1 = activ(conv1d(scan_con_wps, 'c1d_1', n_filters=32, filter_size=5, stride=2, init_scale=np.sqrt(2), **kwargs_conv))
    layer_2 = activ(conv1d(layer_1, 'c1d_2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs_conv))
    layer_2 = conv_to_fc(layer_2)
    layer_3 = activ(linear(layer_2, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    # print_goal = tf.print(wps)
    # with tf.control_dependencies([print_goal]):
    layer_4 = activ(linear(layer_3, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))
    return layer_4


class CNN1DPolicy(common.FeedForwardPolicy):
    """
    This class provides a 1D convolutional network for the Polar Representation
    """
    def __init__(self, *args, **kwargs):
        super(CNN1DPolicy, self).__init__(*args, **kwargs, cnn_extractor=laser_cnn_stack, feature_extraction="cnn")


########################################
###### CnnPolicy_multi_input_vel #######
########################################
def nature_cnn_multi_input(state, **kwargs):
    """
    CNN from Nature paper. state[0:-1, :, :] is the input image, while state[-1, :, :]
    provides additional information, that is concenated later with the output pf layer 3.
    It can be additional non-image information.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    img = state[:, 0:-1 , : , :]
    vel = state[:, -1, 0:kwargs['multi_input_size'], 0]
    vel = tf.math.multiply(vel, kwargs['max_image_value'])
    kwargs_conv = {}
    activ = tf.nn.relu
    layer_1 = activ(conv(img, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs_conv))

    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs_conv))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs_conv))
    layer_3 = conv_to_fc(layer_3)
    concat_img_goal = tf.concat([layer_3, vel], 1)
    # For printing uncomment: velocity
    # print_vel = tf.print(vel)
    # with tf.control_dependencies([print_vel]):
    layer_4 = activ(linear(concat_img_goal, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
    return layer_4


class CnnPolicy_multi_input_vel(common.FeedForwardPolicy):
    """
    This class provides a 2D convolutional network for the X-Image Speed Representation
    """
    def __init__(self, *args, **kwargs):
        kwargs["multi_input_size"] = 2
        kwargs["max_image_value"] = 100
        super(CnnPolicy_multi_input_vel, self).__init__(*args, **kwargs, cnn_extractor=nature_cnn_multi_input, feature_extraction="cnn")




#########################################
###### CnnPolicy_multi_input_vel2 #######
#########################################
def cnn_multi_input(state, **kwargs):
    """
    CNN from Nature paper. state[0:-1, :, :] is the input image, while state[-1, :, :]
    provides additional information, that is concenated later with the output pf layer 3.
    It can be additional non-image information.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    img = state[:, 0:-1 , : , :]
    vel = state[:, -1, 0:kwargs['multi_input_size'], -1]
    vel = tf.math.multiply(vel, kwargs['max_image_value'])
    kwargs_conv = {}
    activ = tf.nn.relu
    layer_1 = activ(conv(img, 'c1', n_filters=64, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs_conv))

    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs_conv))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs_conv))
    layer_4 = activ(conv(layer_3, 'c4', n_filters=32, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs_conv))
    layer_4 = conv_to_fc(layer_4)
    concat_img_goal = tf.concat([layer_4, vel], 1)
    # For printing uncomment: velocity
    # print_vel = tf.print(vel)
    # with tf.control_dependencies([print_vel]):
    layer_4 = activ(linear(concat_img_goal, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
    layer_5 = activ(linear(layer_4, 'fc2', n_hidden=216, init_scale=np.sqrt(2)))
    return layer_5


class CnnPolicy_multi_input_vel2(common.FeedForwardPolicy):
    """
    This class provides a 2D convolutional network for the X-Image Speed Representation
    """
    def __init__(self, *args, **kwargs):
        kwargs["multi_input_size"] = 2
        kwargs["max_image_value"] = 100
        super(CnnPolicy_multi_input_vel2, self).__init__(*args, **kwargs, cnn_extractor=cnn_multi_input, feature_extraction="cnn")


# Deprecated
class CnnPolicy_multi_input_vel3(common.FeedForwardPolicy):
    """
    CnnPolicy_multi_input_vel3 == CnnPolicy_multi_input_vel2
    Is needed because some agents are trained on this policy-name and is still needed for execution mode
    """
    def __init__(self, *args, **kwargs):
        kwargs["multi_input_size"] = 2
        kwargs["max_image_value"] = 100
        super(CnnPolicy_multi_input_vel3, self).__init__(*args, **kwargs, cnn_extractor=cnn_multi_input, feature_extraction="cnn")

