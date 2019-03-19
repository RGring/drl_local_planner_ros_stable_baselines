#ifndef SAFETY_STOP_H
#define SAFETY_STOP_H
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud.h>
#include <laser_geometry/laser_geometry.h>
#include <tf/transform_listener.h>



namespace rl_local_planner {

  /**
   * @class RLLocalPlannerDummy
   * @brief Dummy Planner to the BaseLocalPlanner interface to explore interfaces and integration in move_base.
   */
  class SafetyStop {
    public:
      /**
       * @brief  Constructor for MIRPlannerROS wrapper
       */
      SafetyStop(const ros::NodeHandle& node_handle, double thresh);

      
      virtual ~SafetyStop() = default;

      bool is_safe();
      double get_thesh();

    protected:
      ros::NodeHandle nh_;
    private:
      ros::Subscriber sub_f_scan;
      ros::Subscriber sub_b_scan;
      bool safe_;
      bool b_safe_;
      bool f_safe_;
      double thresh_;
      tf::TransformListener listener_;
      laser_geometry::LaserProjection projector_;
      void f_scan_callback(const sensor_msgs::LaserScanConstPtr& laser);
      void b_scan_callback(const sensor_msgs::LaserScanConstPtr& laser);
      bool is_scan_safe(const std::vector<geometry_msgs::Point32> vec);
      sensor_msgs::PointCloud laserToCloud(const sensor_msgs::LaserScanConstPtr& laser,
                                           std::string target_frame);

      void set_safe();


  };
};
#endif /* SAFETY_STOP_H */
