/*
 * @name	 	  laser_scan_merger.h
 * @brief	 	  Merges several scans to a single one with coordinate system in robot_frame
 * @author  	Ronja Gueldenring
 * @date 	  	2019/04/05
 * @comment 	Code snippets are taken from: https://github.com/iralabdisco/ira_laser_tools
 **/

#ifndef LASER_SCAN_MERGER_H
#define LASER_SCAN_MERGER_H

#include <rl_msgs/MergeScans.h>
#include <ros/ros.h>
#include <laser_geometry/laser_geometry.h>
#include <tf/transform_listener.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h> 
#include <pcl/conversions.h>
#include <sensor_msgs/PointCloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/point_cloud2_iterator.h>

namespace rl_agent {
  /**
   * This class merges several laserscan to one single laser scan.
   */
  class LaserScanMerger{
    public:
      LaserScanMerger(const ros::NodeHandle& node_handle);
    private:
    ros::NodeHandle nh_;
    laser_geometry::LaserProjection projector_;
    tf::TransformListener listener_;

    /**
     * 
     * @brief Generates a Laserscan from a pointcloud
     * 
     **/ 
    sensor_msgs::LaserScanPtr pointcloud_to_laserscan(Eigen::MatrixXf points, pcl::PCLPointCloud2 *merged_cloud);

    /**
     * merge_service_ + merge_scan_callback
     * @brief Merges several laserscan to one.
     * 
     **/ 
    ros::ServiceServer merge_service_;
    bool merge_scan_callback(rl_msgs::MergeScans::Request& request, rl_msgs::MergeScans::Response& response);
    
    double angle_increment_, time_increment_, range_min_, range_max_, scan_time_;
    double max_height_, min_height_;
    std::string robot_frame_;
  };
};
#endif /* LASER_SCAN_MERGER_H */
