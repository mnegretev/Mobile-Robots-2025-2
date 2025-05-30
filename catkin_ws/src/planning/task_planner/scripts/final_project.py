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

NAME = "Miguel Angel Ruiz Sànchez"

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


def calculate_inverse_kinematics(x, y, z, roll, pitch, yaw, arm='left'):
    try:
        service_name = "/manipulation/la_ik_trajectory" if arm == 'left' else "/manipulation/ra_ik_trajectory"
        
        req = InverseKinematicsPose2TrajRequest()
        req.x = x
        req.y = y
        req.z = z
        req.roll = roll
        req.pitch = pitch
        req.yaw = yaw
        req.duration = 0;
        req.time_step = 0.05
        
        
        
        req.initial_guess = []
        clt = rospy.ServiceProxy(service_name, InverseKinematicsPose2Traj)
        resp = clt(req)
        return resp.articular_trajectory
    except Exception as e:
        rospy.logerr(f"Error en cinemática inversa ({arm} arm): {str(e)}")
        return None
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
def get_ra_polynomial_trajectory(q, duration=5.0, time_step=0.05):
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
def get_polynomial_trajectory(q, arm='left', duration=2.0, time_step=0.05):
    topic_name = "/hardware/left_arm/current_pose" if arm == 'left' else "/hardware/right_arm/current_pose"
    current_p = rospy.wait_for_message(topic_name, Float64MultiArray)
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
# the xyz coordinates of the requested object w.r.t. "kinect_link"
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
def transform_point(x, y, z, source_frame="kinect_link", target_frame="shoulders_left_link"):
    print("ENTRO AL transform")
    try:
        
        listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(3.0))
        
        obj_p = PointStamped()
        obj_p.header.frame_id = source_frame
        obj_p.header.stamp = rospy.Time(0)
        obj_p.point.x = x
        obj_p.point.y = y
        obj_p.point.z = z
        
        
        obj_p = listener.transformPoint(target_frame, obj_p)
        return [obj_p.point.x, obj_p.point.y, obj_p.point.z]
    
    except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr("TF transform error: %s", e)
        print(e)
        return None

def main():
    global new_task, recognized_speech, executing_task, goal_reached
    global pubLaGoalPose, pubRaGoalPose, pubHdGoalPose, pubLaGoalGrip, pubRaGoalGrip
    global pubLaGoalTraj, pubRaGoalTraj, pubGoalPose, pubCmdVel, pubSay, listener
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
    listener = tf.TransformListener()
    print("Waiting for services...")
    rospy.wait_for_service('/manipulation/la_ik_pose')
    rospy.wait_for_service('/manipulation/ra_ik_pose')
    rospy.wait_for_service('/vision/obj_reco/detect_and_recognize_object')
    print("Services are now available.")
    loop = rospy.Rate(10)
    

    #
    # FINAL PROJECT 
    #
    executing_task = False
    current_state = "SM_INIT"
    new_task = False
    goal_reached = False
    recognized_speech = ""
    object_to_grab = ""
    destination = []
    desk_near_pose = [3.5, 5.8]  
    object_position = [0, 0, 0]
    home_position_left = [0,0,0,0,0,0,0]  
    home_position_right = [0,0,0,0,0,0,0]
    arm_in_use = None
    pregrasp_lift = None
    pregrasp_extend = None
    grasp_position = None
    say("Ready")
    while not rospy.is_shutdown():
        #
        # Write here your AFSM
        #
        if current_state == "SM_INIT":
            print("Iniciando máquina de estados")
            say("Hello. I'm ready to execute a command.")
            move_head(0, 0)  
            move_left_arm(0,0,0,0,0,0,0)  
            move_left_gripper(0)   
            move_right_arm(0,0,0,0,0,0,0)
            move_right_gripper(0)
            current_state = "SM_WAIT_FOR_COMMAND"

        elif current_state == "SM_WAIT_FOR_COMMAND":
            if new_task:
                new_task = False
                current_state = "SM_PARSE_COMMAND"
            #current_state = "SM_PARSE_COMMAND"


        elif current_state == "SM_PARSE_COMMAND":
            say("I heard: " + recognized_speech)
            object_to_grab, destination = parse_command(recognized_speech)
            say("I will take the " + object_to_grab + " to the " + ("table" if destination == [8.0,8.5] else "kitchen"))
            current_state = "SM_NAVIGATE_NEAR_DESK"

        elif current_state == "SM_NAVIGATE_NEAR_DESK":
            say("Navigating near the desk")
            go_to_goal_pose(desk_near_pose[0], desk_near_pose[1])
            goal_reached = False
            current_state = "SM_NAVIGATE_NEAR_DESK_WAIT"

        elif current_state == "SM_NAVIGATE_NEAR_DESK_WAIT":
            if goal_reached:
                say("I arrived near the desk")
                current_state = "SM_APPROACH_DESK"

        elif current_state == "SM_APPROACH_DESK":
            say("Approaching the desk")
            move_base(0.0, 0.21, 3)
            move_base(0.05, 0, 1)   
            current_state = "SM_FIND_OBJECT"

        elif current_state == "SM_FIND_OBJECT":
            say("Looking for the " + object_to_grab)
            move_head(0, -1.7)   
            time.sleep(1)
            
            try:
                obj_cam = find_object(object_to_grab)
                print("obj_pos: ", obj_cam)
                if obj_cam is None:
                    raise Exception("Object not found")
                    
                object_position = obj_cam
                current_state = "SM_DECIDE_ARM"
            except:
                say("I could not find the object. I will try again")
                current_state = "SM_WAIT_FOR_COMMAND"
        
        elif current_state == "SM_DECIDE_ARM":
            try:
                # Transformar a coordenadas del hombro izquierdo
                obj_left = transform_point(object_position[0], object_position[1], object_position[2],
                                          source_frame="kinect_link", 
                                          target_frame="shoulders_left_link")
                print("obj_pos_left: ", obj_left)
                if obj_left is None:
                    raise Exception("Transformation to left shoulder failed")
                
                # Transformar a coordenadas del hombro derecho
                obj_right = transform_point(object_position[0], object_position[1], object_position[2],
                                          source_frame="kinect_link", 
                                          target_frame="shoulders_right_link")
                print("obj_pos_right: ", obj_right)
                if obj_right is None:
                    raise Exception("Transformation to right shoulder failed")
                
                # Decidir qué brazo usar basado en la posición del objeto
                if abs(obj_left[1]) < 0.3:  
                    arm_in_use = 'left' if obj_left[1] > 0 else 'right'
                else:
                    arm_in_use = 'left' if obj_left[1] > 0 and obj_left[1] < obj_right[1] else 'right'
                
                say("I will use my " + arm_in_use + " arm")
                current_state = "SM_PREGRASP_" + arm_in_use.upper()
                
            except Exception as e:
                rospy.logerr("Arm decision failed: " + str(e))
                say("I could not decide which arm to use. Trying left arm by default.")
                arm_in_use = 'left'
                current_state = "SM_PREGRASP_LEFT"

        elif current_state == "SM_PREGRASP_LEFT":
            say("Preparing left arm")
            try:
                obj_shoulder = transform_point(object_position[0], object_position[1], object_position[2],
                                          source_frame="kinect_link", 
                                          target_frame="shoulders_left_link")
                print("obj_pos_left: ", obj_shoulder)
                if obj_shoulder is None:
                    raise Exception("Transformation to left shoulder failed")
                
                # Primero: Levantar el brazo (evitar extensión)
                move_left_arm(0, -0.5, -1.0, 0.5, 0, 0, 0)  
                # Segundo: Posición pregrasp elevada
                pregrasp_lift = [obj_shoulder[0]+0.05, obj_shoulder[1], obj_shoulder[2] + 0.15]  
                traj_pre = calculate_inverse_kinematics(pregrasp_lift[0], pregrasp_lift[1], pregrasp_lift[2], 
                                                        0, -1.57, 0, arm='left')
                if traj_pre is None:
                    raise Exception("IK for left arm pregrasp lift failed")
                move_left_gripper(0.5)
                move_left_arm_with_trajectory(traj_pre)
                current_state = "SM_GRASP_EXTEND_LEFT"
                
            except Exception as e:
                rospy.logerr("Left arm pregrasp failed: " + str(e))
                say("Cannot use left arm. Trying right arm.")
                current_state = "SM_PREGRASP_RIGHT"



        elif current_state == "SM_PREGRASP_RIGHT":
            say("Preparing right arm")
            try:
                obj_shoulder = transform_point(object_position[0], object_position[1], object_position[2],
                                          source_frame="kinect_link", 
                                          target_frame="shoulders_right_link")
                print("obj_pos_right: ", obj_shoulder)
                if obj_shoulder is None:
                    raise Exception("Transformation to right shoulder failed")
                
                # Primero: Levantar el brazo
                move_right_arm(0, -0.5, -1.0, 0.5, 0, 0, 0) 
                
                # Segundo: Posición pregrasp elevada
                pregrasp_lift = [obj_shoulder[0]+0.05, obj_shoulder[1], obj_shoulder[2] + 0.15]  
                traj_pre = calculate_inverse_kinematics(pregrasp_lift[0], pregrasp_lift[1], pregrasp_lift[2], 
                                                        0, -1.57, 0, arm='right')
                if traj_pre is None:
                    raise Exception("IK for right arm pregrasp lift failed")
                move_right_gripper(0.5)
                move_right_arm_with_trajectory(traj_pre)
                current_state = "SM_GRASP_OBJECT"
                
            except Exception as e:
                rospy.logerr("Right arm pregrasp failed: " + str(e))
                say("Cannot grasp the object with either arm.")
                executing_task = False
                move_left_arm(*home_position_left)
                move_right_arm(*home_position_right)
                current_state = "SM_INIT"

        

        elif current_state == "SM_GRASP_OBJECT":
            say("Grabbing the object")
            try:
                if arm_in_use == 'left':
                    move_left_gripper(-0.5)  
                else:
                    move_right_gripper(-0.5) 
                
                time.sleep(1)
                current_state = "SM_NAVIGATE_TO_DESTINATION"
                
            except Exception as e:
                rospy.logerr("Grasping failed: " + str(e))
                say("Failed to grasp the object")
                current_state = "SM_NAVIGATE_TO_DESTINATION" 

        

        elif current_state == "SM_NAVIGATE_TO_DESTINATION":
            say("Navigating to the destination")
            go_to_goal_pose(destination[0], destination[1])
            goal_reached = False
            current_state = "SM_NAVIGATE_TO_DESTINATION_WAIT"

        elif current_state == "SM_NAVIGATE_TO_DESTINATION_WAIT":
            if goal_reached:
                say("I arrived at the destination")
                current_state = "SM_RELEASE_OBJECT"

        elif current_state == "SM_RELEASE_OBJECT":
            say("Releasing the object")
            if arm_in_use == 'left':
                move_left_gripper(0)  
                move_left_arm(0,0,0,0,0,0,0)  
            else:
                move_right_gripper(0)  
                move_right_arm(0,0,0,0,0,0,0) 
            move_head(0, 0)  
            say("Task finished")
            current_state = "SM_INIT"

        loop.sleep()

if __name__ == '__main__':
    main()
