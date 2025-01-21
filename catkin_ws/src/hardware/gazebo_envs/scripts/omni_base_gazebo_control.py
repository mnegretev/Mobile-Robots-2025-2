#!/usr/bin/env python3
import rospy
import math
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import *
import tf

RATE = 100
WHEEL_RADIUS = 0.08
BASE_RADIUS  = 0.23
ALPHA_LEFT   =  math.pi/3
ALPHA_RIGHT  = -math.pi/3
ALPHA_BACK   =  math.pi

def callback_twist(msg):
    global goal_vx, goal_vy, goal_va
    goal_vx = msg.linear.x
    goal_vy = msg.linear.y
    goal_va = msg.angular.z

def main():
    global goal_vx, goal_vy, goal_va
    print("INITIALIZING OMNI BASE GAZEBO CONTROL BY MARCOSOFT...")
    rospy.init_node("omni_base_gazebo_control")
    rospy.Subscriber("/cmd_vel", Twist, callback_twist)
    rospy.wait_for_service("/gazebo/apply_joint_effort")
    rospy.wait_for_service("/gazebo/get_joint_properties")
    rospy.wait_for_service("/gazebo/get_model_state")
    clt_apply_effort   = rospy.ServiceProxy("/gazebo/apply_joint_effort", ApplyJointEffort)
    clt_get_properties = rospy.ServiceProxy("/gazebo/get_joint_properties", GetJointProperties)
    clt_model_state    = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    br = tf.TransformBroadcaster()
    req_effort_left  = ApplyJointEffortRequest()
    req_effort_right = ApplyJointEffortRequest()
    req_effort_back  = ApplyJointEffortRequest()
    req_effort_left.joint_name  = 'justina::wheel_left_joint'
    req_effort_right.joint_name = 'justina::wheel_right_joint'
    req_effort_back.joint_name  = 'justina::wheel_back_joint'
    req_effort_left.duration  = rospy.Duration.from_sec(1.0/RATE)
    req_effort_right.duration = rospy.Duration.from_sec(1.0/RATE)
    req_effort_back.duration  = rospy.Duration.from_sec(1.0/RATE)
    req_prop_left  = GetJointPropertiesRequest()
    req_prop_right = GetJointPropertiesRequest()
    req_prop_back  = GetJointPropertiesRequest()
    req_prop_left.joint_name  = 'justina::wheel_left_joint'
    req_prop_right.joint_name = 'justina::wheel_right_joint'
    req_prop_back.joint_name  = 'justina::wheel_back_joint'
    loop = rospy.Rate(RATE)
    
    req_model_state = GetModelStateRequest()
    req_model_state.model_name = 'justina'
    resp_model_state = clt_model_state(req_model_state)
    robot_x, robot_y, robot_a = 0,0,0
    
    goal_vx, goal_vy, goal_va = 0,0,0
    last_error_left  = 0
    last_error_right = 0
    last_error_back  = 0
    s_alpha_left  = math.sin(ALPHA_LEFT)
    s_alpha_right = math.sin(ALPHA_RIGHT)
    s_alpha_back  = math.sin(ALPHA_BACK)
    c_alpha_left  = math.cos(ALPHA_LEFT)
    c_alpha_right = math.cos(ALPHA_RIGHT)
    c_alpha_back  = math.cos(ALPHA_BACK)

    services_ready = False
    while not services_ready:
        try:
            resp_left  = clt_get_properties(req_prop_left)
            resp_right = clt_get_properties(req_prop_right)
            resp_back  = clt_get_properties(req_prop_back)
            current_left, current_right, current_back = resp_left.rate[0], resp_right.rate[0], resp_back.rate[0]
            resp_model_state = clt_model_state(req_model_state)
            xp = resp_model_state.twist.linear.x
            yp = resp_model_state.twist.linear.y
            ap = resp_model_state.twist.angular.z
            services_ready = True
        except:
            services_ready = False
            pass
        loop.sleep()
    print("OmniBaseControl.->Base control for gazebo omni base is ready.")
        
    
    while not rospy.is_shutdown():
        # Get current wheel speeds
        resp_left  = clt_get_properties(req_prop_left)
        resp_right = clt_get_properties(req_prop_right)
        resp_back  = clt_get_properties(req_prop_back)
        current_left, current_right, current_back = resp_left.rate[0], resp_right.rate[0], resp_back.rate[0]
        # Calculate goal speed from cmd_vel
        goal_left  = -(-s_alpha_left *goal_vx + c_alpha_left *goal_vy + BASE_RADIUS*goal_va)/WHEEL_RADIUS
        goal_right = -(-s_alpha_right*goal_vx + c_alpha_right*goal_vy + BASE_RADIUS*goal_va)/WHEEL_RADIUS
        goal_back  = -(-s_alpha_back *goal_vx + c_alpha_back *goal_vy + BASE_RADIUS*goal_va)/WHEEL_RADIUS
        # Calculate error and derivative
        error_left  = goal_left  - current_left
        error_right = goal_right - current_right
        error_back  = goal_back  - current_back
        d_error_left  = error_left  - last_error_left
        d_error_right = error_right - last_error_right
        d_error_back  = error_back  - last_error_back
        last_error_left  = error_left  
        last_error_right = error_right 
        last_error_back  = error_back
        # Calculate and send PD control laws
        req_effort_left.effort  = 2*goal_left +  4.5*(goal_left  - current_left ) + 0.1*d_error_left
        req_effort_right.effort = 2*goal_right + 4.5*(goal_right - current_right) + 0.1*d_error_right
        req_effort_back.effort  = 2*goal_back  + 4.5*(goal_back  - current_back ) + 0.1*d_error_back
        clt_apply_effort(req_effort_left)
        clt_apply_effort(req_effort_right)
        clt_apply_effort(req_effort_back)
        #Get current linear and angular robot speed to integrate and calculate odometry.
        resp_model_state = clt_model_state(req_model_state)
        robot_x  = resp_model_state.pose.position.x
        robot_y  = resp_model_state.pose.position.y
        robot_qx = resp_model_state.pose.orientation.x
        robot_qy = resp_model_state.pose.orientation.y
        robot_qz = resp_model_state.pose.orientation.z
        robot_qw = resp_model_state.pose.orientation.w
        br.sendTransform((robot_x, robot_y, 0), (robot_qx, robot_qy, robot_qz, robot_qw), rospy.Time.now(), "base_link", "odom")
        loop.sleep()


if __name__ == '__main__':
    main()
    
