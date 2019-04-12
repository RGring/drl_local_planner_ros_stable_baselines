'''
    @name:      state_collector.py
    @brief:     This class provides custom policies used in stable-baselines library
    @author:    Ronja Gueldenring
    @version:   3.5
    @date:      2019/04/05
'''

from stable_baselines.common.policies import laser_cnn, laser_cnn_multi_input, nature_cnn_multi_input, cnn_multi_input
import stable_baselines.common.policies as common
import rospy

NS="sim1"

class CNN1DPolicy(common.FeedForwardPolicy):
    """
    This class provides a 1D convolutional network for the Polar Representation
    """
    def __init__(self, *args, **kwargs):
        super(CNN1DPolicy, self).__init__(*args, **kwargs, cnn_extractor=laser_cnn, feature_extraction="cnn")

class CNN1DPolicy_multi_input(common.FeedForwardPolicy):
    """
    This class provides a 1D convolutional network for the Raw Data Representation
    """
    def __init__(self, *args, **kwargs):
        kwargs["laser_scan_len"] = rospy.get_param("%s/rl_agent/scan_size"%NS, 360)
        super(CNN1DPolicy_multi_input, self).__init__(*args, **kwargs, cnn_extractor=laser_cnn_multi_input, feature_extraction="cnn")

class CnnPolicy_multi_input_vel(common.FeedForwardPolicy):
    """
    This class provides a 2D convolutional network for the X-Image Speed Representation
    """
    def __init__(self, *args, **kwargs):
        kwargs["multi_input_size"] = 2
        kwargs["max_image_value"] = 100
        super(CnnPolicy_multi_input_vel, self).__init__(*args, **kwargs, cnn_extractor=nature_cnn_multi_input, feature_extraction="cnn")

class CnnPolicy_multi_input_vel2(common.FeedForwardPolicy):
    """
    This class provides a 2D convolutional network for the X-Image Speed Representation
    """
    def __init__(self, *args, **kwargs):
        kwargs["multi_input_size"] = 2
        kwargs["max_image_value"] = 100
        super(CnnPolicy_multi_input_vel2, self).__init__(*args, **kwargs, cnn_extractor=cnn_multi_input, feature_extraction="cnn")

class CnnPolicy_multi_input_vel3(common.FeedForwardPolicy):
    """
    CnnPolicy_multi_input_vel3 == CnnPolicy_multi_input_vel2
    Is needed because some agents are trained on this policy-name and is still needed for execution mode
    """
    def __init__(self, *args, **kwargs):
        kwargs["multi_input_size"] = 2
        kwargs["max_image_value"] = 100
        super(CnnPolicy_multi_input_vel3, self).__init__(*args, **kwargs, cnn_extractor=cnn_multi_input, feature_extraction="cnn")
