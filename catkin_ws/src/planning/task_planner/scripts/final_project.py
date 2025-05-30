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

rospy.set_param("/tf2_buffer_server/ignore_repeated_tfs", True)

NAME = "Frausto Martinez Juan Carlos"

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
    print("HELLLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

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
    
    print(type(req_ik))
    print(req_ik)
    
    resp = clt(req_ik)
    
    return resp.articular_trajectory

#
# Calls the service for calculating a polynomial trajectory for the left arm
#
def get_polynomial_trajectory_left(q, duration=2.0, time_step=0.05):
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
def get_polynomial_trajectory_right(q, duration=5.0, time_step=0.05):
    current_p = rospy.wait_for_message("/hardware/right_arm/current_pose", Float64MultiArray)
    current_p = current_p.data
    clt = rospy.ServiceProxy("/manipulation/polynomial_trajectory", GetPolynomialTrajectory)
    req = GetPolynomialTrajectoryRequest()
    req.p1 = current_p
    req.p2 = q
    req.duration = duration
    req.time_step = time_step
    
    print("current_p:", current_p)
    print("goal    p:", q)
    
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
    listener = tf.TransformListener()
    listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
    obj_p = PointStamped()
    obj_p.header.frame_id = source_frame
    obj_p.header.stamp = rospy.Time(0)
    obj_p.point.x, obj_p.point.y, obj_p.point.z = x,y,z
    obj_p = listener.transformPoint(target_frame, obj_p)
    return [obj_p.point.x, obj_p.point.y, obj_p.point.z]

def main():
    rospy.set_param("/tf2_buffer_server/ignore_repeated_tfs", True)
    global new_task, recognized_speech, executing_task, goal_reached
    global pubLaGoalPose, pubRaGoalPose, pubHdGoalPose, pubLaGoalGrip, pubRaGoalGrip
    global pubLaGoalTraj, pubRaGoalTraj, pubGoalPose, pubCmdVel, pubSay

    print("FINAL PROJECT - " + NAME)
    rospy.init_node("final_project")

    # ────────────────────  pubs / subs  ────────────────────
    rospy.Subscriber('/hri/sp_rec/recognized', RecognizedSpeech, callback_recognized_speech)
    rospy.Subscriber('/navigation/goal_reached',        Bool,   callback_goal_reached)

    pubGoalPose   = rospy.Publisher('/move_base_simple/goal',          PoseStamped, queue_size=10)
    pubCmdVel     = rospy.Publisher('/cmd_vel',                        Twist,       queue_size=10)
    pubSay        = rospy.Publisher('/hri/speech_generator',           SoundRequest,queue_size= 1)

    pubLaGoalPose = rospy.Publisher("/hardware/left_arm/goal_pose",    Float64MultiArray, queue_size=10)
    pubRaGoalPose = rospy.Publisher("/hardware/right_arm/goal_pose",   Float64MultiArray, queue_size=10)
    pubHdGoalPose = rospy.Publisher("/hardware/head/goal_pose",        Float64MultiArray, queue_size=10)
    pubLaGoalGrip = rospy.Publisher("/hardware/left_arm/goal_gripper", Float64,          queue_size=10)
    pubRaGoalGrip = rospy.Publisher("/hardware/right_arm/goal_gripper",Float64,          queue_size=10)
    pubLaGoalTraj = rospy.Publisher("/manipulation/la_q_trajectory",   JointTrajectory,   queue_size=10)
    pubRaGoalTraj = rospy.Publisher("/manipulation/ra_q_trajectory",   JointTrajectory,   queue_size=10)

    # ────────────────────  Espera de servicios  ────────────────────
    print("Waiting for services…")
    rospy.wait_for_service('/manipulation/la_ik_pose')
    rospy.wait_for_service('/manipulation/ra_ik_pose')
    rospy.wait_for_service('/vision/obj_reco/detect_and_recognize_object')
    print("Services are now available.")

    # ────────────────────  Variables de la FSM  ────────────────────
    executing_task   = False
    current_state    = "SM_INIT"
    new_task         = False
    recognized_speech= ""
    goal_reached     = False

    # Poses predefinidas
    DESK_POSE      = [3.25, 5.70]          # donde están los objetos
    DESK_APPROACH = [3.25, 7.30]          # posición de aproximación
    TABLE_POSE     = [8.00, 8.50]
    KITCHEN_POSE   = [3.22, 9.72]
    HOME_POSE      = [0.00, 0.00]          # “base” del robot

    # Variables que se llenan con cada orden
    target_object    = ""
    target_location  = []                  # [x,y]
    location_name    = ""
    arm              = ""                  # 'left' / 'right'
    obj_cartesian    = []                  # XYZ en realsense_link
    obj_shoulder     = []                  # XYZ transformado
    tries            = 0                   # reintentos de búsqueda

    loop = rospy.Rate(10)                  # 10 Hz
    # ────────────────────  AFSM  ────────────────────
    while not rospy.is_shutdown():

        # ===== SM_INIT =====
        if current_state == "SM_INIT":
            say("System ready")
            move_head(0, 0)
            move_left_arm(0,0,0,0,0,0,0)
            move_right_arm(0,0,0,0,0,0,0)
            executing_task = False
            current_state  = "SM_WAIT_FOR_COMMAND"

        # ===== SM_WAIT_FOR_COMMAND =====
        elif current_state == "SM_WAIT_FOR_COMMAND":
            if new_task and not executing_task:
                executing_task = True
                new_task       = False

                # parseo de la orden
                target_object, target_location = parse_command(recognized_speech)
                location_name = "table" if target_location == TABLE_POSE else "kitchen"
                arm = "left" if target_object == "pringles" else "right"

                say(f"I will take the {target_object} to the {location_name}")
                current_state = "SM_NAVIGATE_TO_APPROACH"

        # ===== SM_NAVIGATE_TO_DESK =====
        elif current_state == "SM_NAVIGATE_TO_APPROACH":
            say("Navigating to the desk")
            go_to_goal_pose(DESK_APPROACH[0], DESK_APPROACH[1])
            goal_reached  = False
            current_state = "SM_WAIT_NAV_APPROACH"

        elif current_state == "SM_WAIT_NAV_APPROACH":
            if goal_reached:
                goal_reached  = False
                current_state = "SM_NAVIGATE_TO_DESK"
                tries = 0
            
        elif current_state == "SM_NAVIGATE_TO_DESK":
            go_to_goal_pose(DESK_POSE[0], DESK_POSE[1])
            goal_reached  = False
            current_state = "SM_WAIT_NAV_DESK"

        elif current_state == "SM_WAIT_NAV_DESK":
            if goal_reached:
                goal_reached  = False
                say("I arrived at the desk")
                move_head(0, -0.8)   # mirar hacia abajo
                current_state = "SM_FIND_OBJECT"
                tries = 0

        # ===== SM_FIND_OBJECT =====
        elif current_state == "SM_FIND_OBJECT":
            try:
                say(f"Looking for the {target_object}")
                obj_cartesian = find_object(target_object)
                frame = "shoulders_left_link" if arm == "left" else "shoulders_right_link"
                obj_shoulder  = transform_point(*obj_cartesian, target_frame=frame) # <─ frame correcto

                current_state = "SM_GRAB_OBJECT"
            except Exception as e:
                print("[WARN] Object not found:", e)
                tries += 1
                if tries >= 3:
                    say("I could not find the object. Cancelling task")
                    current_state = "SM_RESET"
                else:
                    say("Object not found, trying again")
                    time.sleep(1.0)

        # ===== SM_GRAB_OBJECT =====
        elif current_state == "SM_GRAB_OBJECT":
            try:
                if arm == "left":
                    # abrir pinza y colocar brazo pre-agarre
                    move_left_arm(-1.4, 0, 0, 2, 0, 1.16, 0) 
                    move_left_gripper(0.6)

                    grasp_traj = calculate_inverse_kinematics_left(
                        obj_shoulder[0]+0.15, obj_shoulder[1], obj_shoulder[2],
                        0, -math.pi/2, 0)
                    move_left_arm_with_trajectory(grasp_traj)
                    move_left_gripper(-3.0)            # cerrar
                else:  # arm == "right"
                    move_right_arm(-1.4, 0, 0, 2, 0, 1.16, 0)
                    move_right_gripper(0.6)           # abrir

                    grasp_traj = calculate_inverse_kinematics_right(
                        obj_shoulder[0]+0.15, obj_shoulder[1], obj_shoulder[2],
                        0, -math.pi/2, 0)             # idem
                    move_right_arm_with_trajectory(grasp_traj)
                    move_right_gripper(-3.0)   

                say("Object grabbed")
                current_state = "SM_NAVIGATE_TO_DEST"
            except Exception as e:
                print("[ERROR] Grasp failed:", e)
                say("Grasp failed, trying again")
                current_state = "SM_FIND_OBJECT"

        # ===== SM_NAVIGATE_TO_DEST =====
        elif current_state == "SM_NAVIGATE_TO_DEST":
            say(f"Going to the {location_name}")
            go_to_goal_pose(target_location[0], target_location[1])
            goal_reached  = False
            current_state = "SM_WAIT_NAV_DEST"

        elif current_state == "SM_WAIT_NAV_DEST":
            if goal_reached:
                goal_reached  = False
                say(f"I arrived at the {location_name}")
                current_state = "SM_PLACE_OBJECT"

        # ===== SM_PLACE_OBJECT =====
        elif current_state == "SM_PLACE_OBJECT":
            say("Placing the object")
            if arm == "left":
                move_left_gripper(0.6)   # abrir
                time.sleep(1.0)
                move_left_arm(0,0,0,0,0,0,0)
            else:
                move_right_gripper(0.6)
                time.sleep(1.0)
                move_right_arm(0,0,0,0,0,0,0)

            say("Task completed")
            current_state = "SM_RESET"

        # ===== SM_RESET =====
        elif current_state == "SM_RESET":
            move_head(0,0)
            move_left_arm(0,0,0,0,0,0,0)
            move_right_arm(0,0,0,0,0,0,0)
            go_to_goal_pose(HOME_POSE[0], HOME_POSE[1])
            goal_reached  = False
            current_state = "SM_WAIT_NAV_HOME"

        elif current_state == "SM_WAIT_NAV_HOME":
            if goal_reached:
                goal_reached     = False
                executing_task   = False
                say("Ready for next command")
                current_state    = "SM_WAIT_FOR_COMMAND"

        loop.sleep()

if __name__ == "__main__":
    main()
