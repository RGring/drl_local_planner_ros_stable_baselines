/*
 * @name	 	wp_generator.h
 * @brief	 	Generation of waypoints on the global path and 
 * 				 	determination of the closest waypoints to the robot.
 * @author 	Ronja Gueldenring
 * @date 		2019/04/05
 **/

#ifndef WP_GENERATOR_H
#define WP_GENERATOR_H
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <rl_msgs/Waypoint.h>
#include <std_msgs/Bool.h>
#include <rl_msgs/SetPath.h>
#include <iostream>


namespace rl_local_planner {
  /**
   * This class downsamples the global path to waypoints and 
   * determines the next closest waypoints to the robot.
   */
  class WpGenerator {
    public:
      /**
       * @brief  Constructor for WpGenerator wrapper
       */
      WpGenerator(const ros::NodeHandle& node_handle);

      
      virtual ~WpGenerator() = default;

       /**
       * @brief  Determines the next closest waypoints to the robot
       */
      void get_closest_wps();
    protected:
      ros::NodeHandle nh_;
    private:
      std::vector<geometry_msgs::PoseStamped> waypoints_;   //List of all remaining waypoints on path
      double look_ahead_distance_;                          //distance between waypoints
      double wp_reached_dist_;                              //distance, when threshold is reaches
      int num_of_wps_;                                      //number of closest waypoints, that are published
      std::string path_frame_;                              //frame name of path
      std::string robot_frame_;                             //frame name of the robot
      tf::Vector3 goal_;                                    //most recent goal

      ros::Publisher wp_pub_;                               //Publisher for next closest waypoints   
      ros::Publisher wp_reached_pub_;                       //Publisher for reaching waypoint within a wp_reached_dist_
      tf::TransformListener tf_;

      /**
       * @brief global plan is downsampled to a number of waypoints with distance look_ahead_distance_.
       * @param global_plan The most recent global plan.
       */
      void precalculate_waypoints_(std::vector<geometry_msgs::PoseStamped> global_plan);
      
      /**
       * @brief Mean square distance.
       * @param x x-position
       * @param y x-position
       * @retur sqrt(x^2 + y^2)
       */
      double metric_dist(double x, double y);
      tf::Vector3 get_transformed_wp_(geometry_msgs::PoseStamped wp, tf::StampedTransform transform);

      /**
       * set_path_service + path_callback_
       * @brief If new path available, it can be set here.
       * Wp generator only considers most recent path in further computations.
       */
      ros::ServiceServer set_path_service;
	    bool path_callback_(rl_msgs::SetPath::Request& request, rl_msgs::SetPath::Response& response);
     

  };
};
#endif /* WP_GENERATOR_H */
