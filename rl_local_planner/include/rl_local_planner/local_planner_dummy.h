/*********************************************************************
*
* 
* Ronja Gueldenring
*
*
*********************************************************************/

#ifndef RL_LOCAL_PLANNER_DUMMY_H
#define RL_LOCAL_PLANNER_DUMMY_H
#include <rl_local_planner/safety_stop.h>
#include <nav_core/base_local_planner.h>
#include <ros/ros.h>


namespace rl_local_planner {

  /**
   * @class LocalPlannerDummy
   * @brief Dummy Planner to the BaseLocalPlanner interface to explore interfaces and integration in move_base.
   */
  class LocalPlannerDummy : public nav_core::BaseLocalPlanner {
    public:
      /**
       * @brief  Constructor for MIRPlannerROS wrapper
       */
      LocalPlannerDummy();

      
      // virtual ~LocalPlannerDummy() = default;


      /**
       * @brief  Constructs the ros wrapper
       * @param name The name to give this instance of the trajectory planner
       * @param tf A pointer to a transform listener
       * @param costmap The cost map to use for assigning costs to trajectories
       */
      void initialize(std::string name, tf::TransformListener* tf, costmap_2d::Costmap2DROS* costmap_ros);

      /**
       * @brief  Check if the goal pose has been achieved
       * @return True if achieved, false otherwise
       */
      bool isGoalReached();

      /**
       * @brief  Set the plan that the controller is following
       * @param orig_global_plan The plan to pass to the controller
       * @return True if the plan was updated successfully, false otherwise
       */
      bool setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_plan, double goal_threshold);

      /**
       * @brief  Given the current position, orientation, and velocity of the robot, compute velocity commands to send to the base
       * @param cmd_vel Will be filled with the velocity command to be passed to the robot base
       * @return >0 if a valid velocity command was found, otherwise a result <0, contains the fail_code
       */
      int computeVelocityCommands(geometry_msgs::Twist& cmd_vel);
    protected:
      ros::NodeHandle nh_;
    private:
      rl_local_planner::SafetyStop* safety_stop_;
      tf::TransformListener* tf_;
      std::vector<geometry_msgs::PoseStamped> global_plan_;
      tf::Vector3 original_goal_;
      double goal_threshold_;
      std::string path_frame_;
      std::string robot_frame_;
      double look_ahead_distance_;
      tf::Vector3 get_next_waypoint();
      double metric_dist(double x, double y);
  };
};
#endif /* RL_LOCAL_PLANNER_DUMMY_H */
