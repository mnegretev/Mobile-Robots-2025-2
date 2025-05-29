#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# FINAL PROJECT - SIMPLE SERVICE ROBOT
# 
# Instructions:
# Write the code necessary to make the robot to perform the following possible commands:
# * Robot take the <pringles|drink> to the <table|kitchen>
# You can choose where the table and kitchen are located within the map.
# The Robot must recognize the orders using speech recognition.
# Entering the command by text or similar way is not allowed.
# The Robot must announce the progress of the action using speech synthesis,
# for example: I'm going to grab..., I'm going to navigate to ..., I arrived to..., etc.
# Publishers and suscribers to interact with the subsystems (navigation,
# vision, manipulation, speech synthesis and recognition) are already declared. 
#

import rospy
import tf
import numpy as np
import math
import time
from std_msgs.msg import String, Float64MultiArray, Float64, Bool
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan, GetPlanRequest
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point, PointStamped
from trajectory_msgs.msg import JointTrajectory
from sound_play.msg import SoundRequest
from vision_msgs.srv import *
from manip_msgs.srv import *
from hri_msgs.msg import *

NAME = "JORGE EITHAN TREVIÑO SELLES"

listener = None

#
# Global variable 'speech_recognized' contains the last recognized sentence
#
def callback_recognized_speech(msg):
    global recognized_speech, new_task, executing_task
    if executing_task:
        return
    new_task = True
    recognized_speech = msg.hypothesis[0]
    print("New command recognized: " + recognized_speech)

#
# Global variable 'goal_reached' is set True when the last sent navigation goal is reached
#
def callback_goal_reached(msg):
    global goal_reached
    goal_reached = msg.data
    print("Received goal reached: " + str(goal_reached))

def parse_command(cmd):
    obj = "pringles" if "PRINGLES" in cmd else "drink"
    loc = [8.0,8.5] if "TABLE" in cmd else [3.22, 9.72]
    return obj, loc

#
# This function sends the goal articular position to the left arm and sleeps 2 seconds
# to allow the arm to reach the goal position. 
#
def move_left_arm(q1,q2,q3,q4,q5,q6,q7):
    global pubLaGoalPose
    msg = Float64MultiArray()
    msg.data.append(q1)
    msg.data.append(q2)
    msg.data.append(q3)
    msg.data.append(q4)
    msg.data.append(q5)
    msg.data.append(q6)
    msg.data.append(q7)
    pubLaGoalPose.publish(msg)
    time.sleep(2.0)

#
# This function sends and articular trajectory to the left arm and sleeps proportional to length of trajectory
# to allow the arm to reach the goal position. 
#
def move_left_arm_with_trajectory(Q):
    global pubLaGoalTraj
    pubLaGoalTraj.publish(Q)
    time.sleep(0.05*len(Q.points) + 2)

#
# This function sends the goal angular position to the left gripper and sleeps 1 second
# to allow the gripper to reach the goal angle. 
#
def move_left_gripper(q):
    global pubLaGoalGrip
    pubLaGoalGrip.publish(q)
    time.sleep(1.0)

#
# This function sends the goal articular position to the right arm and sleeps 2 seconds
# to allow the arm to reach the goal position. 
#
def move_right_arm(q1,q2,q3,q4,q5,q6,q7):
    global pubRaGoalPose
    msg = Float64MultiArray()
    msg.data.append(q1)
    msg.data.append(q2)
    msg.data.append(q3)
    msg.data.append(q4)
    msg.data.append(q5)
    msg.data.append(q6)
    msg.data.append(q7)
    pubRaGoalPose.publish(msg)
    time.sleep(2.0)

#
# This function sends and articular trajectory to the right arm and sleeps proportional to length of trajectory
# to allow the arm to reach the goal position. 
#
def move_right_arm_with_trajectory(Q):
    global pubRaGoalTraj
    pubRaGoalTraj.publish(Q)
    time.sleep(0.05*len(Q.points) + 2)

#
# This function sends the goal angular position to the right gripper and sleeps 1 second
# to allow the gripper to reach the goal angle. 
#
def move_right_gripper(q):
    global pubRaGoalGrip
    pubRaGoalGrip.publish(q)
    time.sleep(1.0)

#
# This function sends the goal pan-tilt angles to the head and sleeps 1 second
# to allow the head to reach the goal position. 
#
def move_head(pan, tilt):
    global pubHdGoalPose
    msg = Float64MultiArray()
    msg.data.append(pan)
    msg.data.append(tilt)
    pubHdGoalPose.publish(msg)
    time.sleep(1.0)

#
# This function sends a linear and angular speed to the mobile base to perform
# low-level movements. The mobile base will move at the given linear-angular speeds
# during a time given by 't'
#
def move_base(linear, angular, t):
    global pubCmdVel
    cmd = Twist()
    cmd.linear.x = linear
    cmd.angular.z = angular
    pubCmdVel.publish(cmd)
    time.sleep(t)
    pubCmdVel.publish(Twist())

#
# This function publishes a global goal position. This topic is subscribed by
# pratice04 and performs path planning and tracking.
#
def go_to_goal_pose(goal_x, goal_y):
    global pubGoalPose
    goal_pose = PoseStamped()
    goal_pose.pose.orientation.w = 1.0
    goal_pose.pose.position.x = goal_x
    goal_pose.pose.position.y = goal_y
    pubGoalPose.publish(goal_pose)

#
# This function sends a text to be synthetized.
#
def say(text):
    global pubSay
    msg = SoundRequest()
    msg.sound   = -3
    msg.command = 1
    msg.volume  = 1.0
    msg.arg2    = "voice_kal_diphone"
    msg.arg = text
    print("Saying: " + text)
    pubSay.publish(msg)
    time.sleep(len(text)*0.1)

#
# This function calls the service for calculating inverse kinematics for left arm
# and returns the calculated articular position.
#
def calculate_inverse_kinematics_left(x,y,z,roll, pitch, yaw):
    req_ik = InverseKinematicsPose2TrajRequest()
    req_ik.x = x
    req_ik.y = y
    req_ik.z = z
    req_ik.roll  = roll
    req_ik.pitch = pitch
    req_ik.yaw   = yaw
    req_ik.duration = 0;
    req_ik.time_step = 0.05
    req_ik.initial_guess = []
    clt = rospy.ServiceProxy("/manipulation/la_ik_trajectory", InverseKinematicsPose2Traj)
    resp = clt(req_ik)
    return resp.articular_trajectory

#
# This function calls the service for calculating inverse kinematics for right arm
# and returns the calculated articular position.
#
def calculate_inverse_kinematics_right(x,y,z,roll, pitch, yaw):
    req_ik = InverseKinematicsPose2TrajRequest()
    req_ik.x = x
    req_ik.y = y
    req_ik.z = z
    req_ik.roll  = roll
    req_ik.pitch = pitch
    req_ik.yaw   = yaw
    req_ik.duration = 0;
    req_ik.time_step = 0.05
    req_ik.initial_guess = []
    clt = rospy.ServiceProxy("/manipulation/ra_ik_trajectory", InverseKinematicsPose2Traj)
    resp = clt(req_ik)
    return resp.articular_trajectory

#
# Calls the service for calculating a polynomial trajectory for the left arm
#
def get_la_polynomial_trajectory(q, duration=2.0, time_step=0.05):
    current_p = rospy.wait_for_message("/hardware/left_arm/current_pose", Float64MultiArray)
    current_p = current_p.data
    clt = rospy.ServiceProxy("/manipulation/polynomial_trajectory", GetPolynomialTrajectory)
    req = GetPolynomialTrajectoryRequest()
    req.p1 = current_p
    req.p2 = q
    req.duration = duration
    req.time_step = time_step
    resp = clt(req)
    return resp.trajectory

#
# Calls the service for calculating a polynomial trajectory for the right arm
#
def get_la_polynomial_trajectory(q, duration=5.0, time_step=0.05):
    current_p = rospy.wait_for_message("/hardware/right_arm/current_pose", Float64MultiArray)
    current_p = current_p.data
    clt = rospy.ServiceProxy("/manipulation/polynomial_trajectory", GetPolynomialTrajectory)
    req = GetPolynomialTrajectoryRequest()
    req.p1 = current_p
    req.p2 = q
    req.duration = duration
    req.time_step = time_step
    resp = clt(req)
    return resp.trajectory


#
# Calls the service for finding object and returns
# the xyz coordinates of the requested object w.r.t. "realsense_link"
#
def find_object(object_name):
    clt_find_object = rospy.ServiceProxy("/vision/obj_reco/detect_and_recognize_object", RecognizeObject)
    req_find_object = RecognizeObjectRequest()
    req_find_object.point_cloud = rospy.wait_for_message("/camera/depth_registered/points", PointCloud2)
    req_find_object.name  = object_name
    resp = clt_find_object(req_find_object)
    return [resp.recog_object.pose.position.x, resp.recog_object.pose.position.y, resp.recog_object.pose.position.z]

#
# Transforms a point xyz expressed w.r.t. source frame to the target frame
#
def transform_point(x,y,z, source_frame="realsense_link", target_frame="shoulders_left_link"):
    global listener
    # Create a PointStamped object with the source frame and point coordinates
    listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
    # Transform the point from source frame to target frame
    obj_p = PointStamped()
    obj_p.header.frame_id = source_frame
    obj_p.header.stamp = rospy.Time(0)
    obj_p.point.x, obj_p.point.y, obj_p.point.z = x,y,z
    # Transform the point
    obj_p = listener.transformPoint(target_frame, obj_p)
    # Return the transformed point coordinates
    return [obj_p.point.x, obj_p.point.y, obj_p.point.z]

#
# This function retrieves the position of a frame in global coordinates
# 
def get_transform(frame):
    global listener
    try:
        # Wait for the transform to be available
        ([x, y, z], _) = listener.lookupTransform('map', frame, rospy.Time(0))
        # Return the position as a numpy array
        return np.asarray([x, y, z])
    except Exception as e:
        return np.asarray([0,0,0])

#
# This function retrieves the current position of a frame in global coordinates
#
def transform_point_mod(loc, is_left_arm):
    # loc is a list or tuple with x, y, z coordinates
    x, y, z = loc
    # Get the position of the target frame and source frame
    target_frame = "shoulders_left_link" if is_left_arm else "shoulders_right_link"
    source_frame = "kinect_base"
    # Get the transformation from the source frame to the target frame
    source_pos = get_transform(source_frame)
    target_pos = get_transform(target_frame)
    # Calculate the point in global coordinates
    point_no_transform = -np.asarray([x, y, z]) + source_pos
    # Return the transformed point in global coordinates
    return target_pos - point_no_transform

#
# Get robot pose
#
def get_robot_pose():
    global listener
    try:
        # Wait for the transform to be available
        ([x, y, z], [qx,qy,qz,qw]) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
        # Convert quaternion to Euler angles
        return np.asarray([x, y]), 2*math.atan2(qz, qw)
    except Exception as e:
        print(f"Error: {e}")
        return np.asarray([0,0]),0

#
# Rotate to target angle
#
def rotate_to_target(target_angle):
        # Get the current robot pose
        _, rotation = get_robot_pose()
        # Rotate until the robot's is close enough to the target angle
        while abs(rotation - target_angle) > 0.05:
            # Get the current robot pose again
            rotation_normalized = (rotation + math.pi) % (2 * math.pi) - math.pi
            target_normalized = (target_angle + math.pi) % (2 * math.pi) - math.pi
            angle_diff = target_normalized - rotation_normalized

            # Normalize the angle difference to the range [-pi, pi]
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Calculate the angular speed based on the angle difference
            angular_speed = 0.75 if angle_diff > 0 else -0.75
            # Move the base with the calculated angular speed
            move_base(0.0, angular_speed, 0.05)
            # Update the robot pose
            _, rotation = get_robot_pose()

def main():
    global new_task, recognized_speech, executing_task, goal_reached
    global pubLaGoalPose, pubRaGoalPose, pubHdGoalPose, pubLaGoalGrip, pubRaGoalGrip
    global pubLaGoalTraj, pubRaGoalTraj, pubGoalPose, pubCmdVel, pubSay
    global pub_point, listener
    print("FINAL PROJECT - " + NAME)
    rospy.init_node("final_project")
    rospy.Subscriber('/hri/sp_rec/recognized', RecognizedSpeech, callback_recognized_speech)
    rospy.Subscriber('/navigation/goal_reached', Bool, callback_goal_reached)
    pubGoalPose   = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    pubCmdVel     = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    pubSay        = rospy.Publisher('/hri/speech_generator', SoundRequest, queue_size=1)
    pubLaGoalPose = rospy.Publisher("/hardware/left_arm/goal_pose" , Float64MultiArray, queue_size=10);
    pubRaGoalPose = rospy.Publisher("/hardware/right_arm/goal_pose", Float64MultiArray, queue_size=10);
    pubHdGoalPose = rospy.Publisher("/hardware/head/goal_pose"     , Float64MultiArray, queue_size=10);
    pubLaGoalGrip = rospy.Publisher("/hardware/left_arm/goal_gripper" , Float64, queue_size=10);
    pubRaGoalGrip = rospy.Publisher("/hardware/right_arm/goal_gripper", Float64, queue_size=10);
    pubLaGoalTraj = rospy.Publisher("/manipulation/la_q_trajectory", JointTrajectory, queue_size=10)
    pubRaGoalTraj = rospy.Publisher("/manipulation/ra_q_trajectory", JointTrajectory, queue_size=10)
    pub_point = rospy.Publisher('/detected_object', PointStamped, queue_size=10)
    listener = tf.TransformListener()
    print("Waiting for services...")
    rospy.wait_for_service('/manipulation/la_ik_pose')
    rospy.wait_for_service('/manipulation/ra_ik_pose')
    rospy.wait_for_service('/vision/obj_reco/detect_and_recognize_object')
    print("Services are now available.")
    loop = rospy.Rate(10)
    q = []
    arm = "LEFT"  # Default arm, can be changed later

    #
    # FINAL PROJECT 
    #
    executing_task = False
    current_state = "SM_INIT"
    new_task = False
    goal_reached = False
    object_is_close = False
    recognized_speech = ""
    say("Ready")
    while not rospy.is_shutdown():
        # SM_INIT
        if current_state == "SM_INIT":
            print("Iniciando máquina de estados")
            say("Hello. I'm ready to execute a command.")
            # move_head(0, -0.5)
            # move_head(0, 0.5)
            # move_head(0, -0.5)
            # move_head(0, 0.5)
            # move_head(0, -0.5)
            # move_head(0, 0)
            # move_base(0.5, 0.0, 3.0)
            # move_left_arm(-1.0, 0.1 , 0.0, 1.5, 0.0, 1.0, 0.0)
            # move_left_gripper(1.0)
            # move_left_gripper(0.0)
            # move_left_arm(0,0,0,0,0,0,0)
            current_state = "SM_WAITING_CMD"
            
        # SM_WAITING_CMD
        elif current_state == "SM_WAITING_CMD":
            if new_task:
                requested_object, requested_location = parse_command(recognized_speech)
                print("New task received: " + requested_object + " to  " + str(requested_location))
                say("Executing the command, " + recognized_speech)
                current_state = "SM_MOVE_NEAR_TABLE"
                new_task = False
                executing_task = True
                
        #SM_MOVE_NEAR_TABLE
        elif current_state == "SM_MOVE_NEAR_TABLE":
            print("Moving near table")
            say("Moving near the table")
            go_to_goal_pose(3.4, 6.0)
            current_state = "SM_ROTATE"
            
            
        elif current_state == "SM_ROTATE":
            if goal_reached:
                print("Rotating to face the table")
                say("Rotating to face the table")
                rotate_to_target(math.pi*3/2)
                current_state = "SM_APPROACH_OBJECT"
        
        # SM_APPROACH_OBJECT
        elif current_state == "SM_APPROACH_OBJECT":
            if goal_reached:
                goal_reached = False
                print("Approaching object")
                say("Approaching the object")
                go_to_goal_pose(3.4, 5.68)
                current_state = "SM_REFORM_ROTATE"
            
        # SM_REFORM_ROTATE
        elif current_state == "SM_REFORM_ROTATE":
            if goal_reached:
                goal_reached = False
                print("Rotating to face the object")
                say("Rotating to face the object")
                rotate_to_target(math.pi*3/2)
                current_state = "SM_MOVE_HEAD_DOWN"
        
        # SM_MOVE_HEAD_DOWN
        elif current_state == "SM_MOVE_HEAD_DOWN":
            print("Moving head down")
            say("Moving my head down to see the object")
            move_head(0, -1.0)
            current_state = "SM_FIND_OBJECT"
        
        # SM_FIND_OBJECT
        elif current_state == "SM_FIND_OBJECT":
            time.sleep(1.0)  # Allow time for the head to move
            print("Finding object")
            say("Finding the object")
            pos = find_object(requested_object)
            arm = "LEFT" if requested_object == "pringles" else "RIGHT"
            current_state = "SM_PREPARE_ARM"
        
        # SM_PREPARE_ARM
        elif current_state == "SM_PREPARE_ARM":
            print("Preparing arm")
            say("Preparing my " + arm + " arm to grab the object")
            if arm == "LEFT":
                move_left_arm(-0.1432, 0.0, -0.1, 1.8418, 0.0, 0.1695, 0.0)
                move_left_gripper(0.4)
            elif arm == "RIGHT":
                move_right_arm(-0.1432, -0.1, -0.1, 1.8418, 0.0, 0.1695, 0.0)
                move_right_gripper(0.4)
            current_state = "SM_CALCULATE_INVERSE_KINEMATICS"
        
        
        # SM_CALCULATE_INVERSE_KINEMATICS"
        elif current_state == "SM_CALCULATE_INVERSE_KINEMATICS":
            # Wait for 3 seconds to allow the robot to stabilize
            time.sleep(3.0)
            try:
                # Find object position in global coordinates
                say("Transforming the position of the object")
                pos = transform_point_mod(pos, arm == "LEFT")
                print("Transformed position:", pos)
                say("Calculating the trajectory to the object")
                if arm == "LEFT":
                    q = calculate_inverse_kinematics_left(pos[0], pos[1], pos[2], 0.0, 0.0, 0.0)
                    q = get_la_polynomial_trajectory(q, duration=2.0, time_step=0.05)
                elif arm == "RIGHT":
                    q = calculate_inverse_kinematics_right(pos[0], pos[1], pos[2], 0.0, 0.0, 0.0)
                    q = get_la_polynomial_trajectory(q, duration=2.0, time_step=0.05)
                current_state = "SM_MOVE_ARM_TO_OBJECT_TRAJ"
            except Exception as e:
                say("I couldn't calculate the trajectory to the object")
                print("Error calculating inverse kinematics:", e)
                current_state = "SM_REPOSITION"
                
        # SM_REPOSITION
        elif current_state == "SM_REPOSITION":
            print("Repositioning to find the object")
            say("Repositioning to find the object")
            if arm == "LEFT":
                move_left_arm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                move_left_gripper(0.4)
            elif arm == "RIGHT":
                move_right_arm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                move_right_gripper(0.4)
            # Move back and forth to find the object
            move_base(-0.2, 0.0, 1.0)
            move_base(0.2, 0.0, 1.0)
            time.sleep(1.0)  # Allow time for the head to move
            current_state = "SM_FIND_OBJECT"
                
        # SM_MOVE_ARM_TO_OBJECT_TRAJ
        elif current_state == "SM_MOVE_ARM_TO_OBJECT_TRAJ":
            print("Moving arm to object")
            say("Moving my " + arm + " arm to the object")
            if arm == "LEFT":
                move_left_arm_with_trajectory(q)
            elif arm == "RIGHT":
                move_right_arm_with_trajectory(q)
            current_state = "SM_GRAB_OBJECT"
                
        # SM_GRAB_OBJECT
        elif current_state == "SM_GRAB_OBJECT":
            print("Grabbing object")
            say("Grabbing the object")
            if arm == "LEFT":
                move_left_gripper(-0.2)
            elif arm == "RIGHT":
                move_right_gripper(-0.2)
            current_state = "SM_MOVE_HEAD_UP"
            
            
            
        # SM_MOVE_HEAD_UP
        elif current_state == "SM_MOVE_HEAD_UP":
            print("Moving head up")
            say("Moving my head up")
            move_head(0, 0)
            current_state = "SM_MOVE_TO_LOCATION"

        # SM_FIND_OBJECTrequested_object, requested_location = parse_command(recognized_speech)
                            
            
        loop.sleep()

if __name__ == '__main__':
    main()
    
