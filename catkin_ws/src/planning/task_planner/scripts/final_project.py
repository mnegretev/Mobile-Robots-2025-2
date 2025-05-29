#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# FINAL PROJECT - SIMPLE SERVICE ROBOT
#

import rospy
import tf
import math
import time
from std_msgs.msg import String, Float64MultiArray, Float64, Bool
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan, GetPlanRequest
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist, PoseStamped, Point, PointStamped
from trajectory_msgs.msg import JointTrajectory
from sound_play.msg import SoundRequest
from vision_msgs.srv import *
from manip_msgs.srv import *
from hri_msgs.msg import *

NAME = "Efren Rivera, Xavier Suastegui, Carlos Lujan y Alan Camarena"

# ----------------------------------------------------------------------------------------------------
# CALLBACKS
# ----------------------------------------------------------------------------------------------------

def callback_recognized_speech(msg):
    global recognized_speech, new_task, executing_task
    if executing_task:
        return
    new_task = True
    recognized_speech = msg.hypothesis[0]
    print("New command recognized: " + recognized_speech)

def callback_goal_reached(msg):
    global goal_reached
    goal_reached = msg.data
    print("Received goal reached: " + str(goal_reached))

# ----------------------------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------------------------------------

def parse_command(cmd):
    obj = "pringles" if "PRINGLES" in cmd else "drink"
    loc = [8.0, 8.5] if "TABLE" in cmd else [3.22, 9.72]
    return obj, loc

def say(text):
    global pubSay
    msg = SoundRequest()
    msg.sound = -3
    msg.command = 1
    msg.volume = 1.0
    msg.arg2 = "voice_kal_diphone"
    msg.arg = text
    rospy.loginfo("Saying: " + text)
    pubSay.publish(msg)
    time.sleep(len(text) * 0.1)

def go_to_goal_pose(goal_x, goal_y):
    global pubGoalPose
    goal_pose = PoseStamped()
    goal_pose.pose.orientation.w = 1.0
    goal_pose.pose.position.x = goal_x
    goal_pose.pose.position.y = goal_y
    pubGoalPose.publish(goal_pose)

def move_base(linear, angular, t):
    global pubCmdVel
    cmd = Twist()
    cmd.linear.x = linear
    cmd.angular.z = angular
    pubCmdVel.publish(cmd)
    time.sleep(t)
    pubCmdVel.publish(Twist())

def move_head(pan, tilt):
    global pubHdGoalPose
    msg = Float64MultiArray()
    msg.data = [pan, tilt]
    pubHdGoalPose.publish(msg)
    time.sleep(1.0)

# ----------------------------------------------------------------------------------------------------
# ARM CONTROL
# ----------------------------------------------------------------------------------------------------

def move_arm(qs, pub):
    msg = Float64MultiArray()
    msg.data = qs
    pub.publish(msg)
    time.sleep(2.0)

def move_arm_trajectory(Q, pub):
    pub.publish(Q)
    time.sleep(0.05 * len(Q.points) + 2)

def move_gripper(q, pub):
    pub.publish(q)
    time.sleep(1.0)

# ----------------------------------------------------------------------------------------------------
# INVERSE KINEMATICS SERVICES
# ----------------------------------------------------------------------------------------------------

def calculate_inverse_kinematics(x, y, z, roll, pitch, yaw, service_name, initial_guess):
    req_ik = InverseKinematicsPose2TrajRequest()
    req_ik.x, req_ik.y, req_ik.z = x, y, z
    req_ik.roll, req_ik.pitch, req_ik.yaw = roll, pitch, yaw
    req_ik.duration = 0
    req_ik.time_step = 0.05
    req_ik.initial_guess = initial_guess
    clt = rospy.ServiceProxy(service_name, InverseKinematicsPose2Traj)
    try:
        resp = clt(req_ik)
        if not resp.articular_trajectory.points:
            rospy.logwarn(f"IK from {service_name} returned no points.")
        return resp.articular_trajectory
    except rospy.ServiceException as e:
        rospy.logerr(f"IK service call failed: {e}")
        return None

# ----------------------------------------------------------------------------------------------------
# VISION AND TRANSFORM
# ----------------------------------------------------------------------------------------------------

def find_object(object_name):
    clt = rospy.ServiceProxy("/vision/obj_reco/detect_and_recognize_object", RecognizeObject)
    req = RecognizeObjectRequest()
    req.point_cloud = rospy.wait_for_message("/camera/depth_registered/points", PointCloud2)
    req.name = object_name
    resp = clt(req)
    return [resp.recog_object.pose.position.x, resp.recog_object.pose.position.y, resp.recog_object.pose.position.z]

def transform_point(x, y, z, source_frame="kinect_link", target_frame="shoulders_left_link"):
    listener = tf.TransformListener()
    try:
        listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0))
        obj_p = PointStamped()
        obj_p.header.frame_id = source_frame
        obj_p.header.stamp = rospy.Time(0)
        obj_p.point.x, obj_p.point.y, obj_p.point.z = x, y, z
        obj_p_out = listener.transformPoint(target_frame, obj_p)
        return [obj_p_out.point.x, obj_p_out.point.y, obj_p_out.point.z]
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr("TF transform failed: %s", str(e))
        return [None, None, None]

# ----------------------------------------------------------------------------------------------------
# MAIN STATE MACHINE
# ----------------------------------------------------------------------------------------------------

def main():
    global new_task, recognized_speech, executing_task, goal_reached
    global pubLaGoalPose, pubRaGoalPose, pubHdGoalPose, pubLaGoalGrip, pubRaGoalGrip
    global pubLaGoalTraj, pubRaGoalTraj, pubGoalPose, pubCmdVel, pubSay

    rospy.init_node("final_project")
    rospy.loginfo("FINAL PROJECT - " + NAME)

    rospy.Subscriber('/hri/sp_rec/recognized', RecognizedSpeech, callback_recognized_speech)
    rospy.Subscriber('/navigation/goal_reached', Bool, callback_goal_reached)

    # Publishers
    pubGoalPose   = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    pubCmdVel     = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    pubSay        = rospy.Publisher('/hri/speech_generator', SoundRequest, queue_size=1)
    pubLaGoalPose = rospy.Publisher("/hardware/left_arm/goal_pose", Float64MultiArray, queue_size=10)
    pubRaGoalPose = rospy.Publisher("/hardware/right_arm/goal_pose", Float64MultiArray, queue_size=10)
    pubHdGoalPose = rospy.Publisher("/hardware/head/goal_pose", Float64MultiArray, queue_size=10)
    pubLaGoalGrip = rospy.Publisher("/hardware/left_arm/goal_gripper", Float64, queue_size=10)
    pubRaGoalGrip = rospy.Publisher("/hardware/right_arm/goal_gripper", Float64, queue_size=10)
    pubLaGoalTraj = rospy.Publisher("/manipulation/la_q_trajectory", JointTrajectory, queue_size=10)
    pubRaGoalTraj = rospy.Publisher("/manipulation/ra_q_trajectory", JointTrajectory, queue_size=10)

    # Wait for services
    rospy.loginfo("Waiting for services...")
    rospy.wait_for_service('/manipulation/la_ik_trayectory')
    rospy.wait_for_service('/manipulation/ra_ik_trayectory')
    rospy.wait_for_service('/vision/obj_reco/detect_and_recognize_object')
    rospy.loginfo("Services ready.")

    # State machine initialization
    executing_task = False
    current_state = "SM_INIT"
    new_task = False
    goal_reached = False
    recognized_speech = ""
    object_name = ""
    target_location = []
    x, y, z = 0, 0, 0

    say("Ready")

    loop = rospy.Rate(10)

    while not rospy.is_shutdown():
        if current_state == "SM_INIT":
            say("Hello. I'm ready to execute a command.")
            current_state = "SM_Waiting"

        elif current_state == "SM_Waiting":
            if new_task:
                executing_task = True
                say("I heard the command: " + recognized_speech)
                object_name, target_location = parse_command(recognized_speech.upper())
                current_state = "SM_ReachTable"

        elif current_state == "SM_ReachTable":
            say("Reaching the table.")
            go_to_goal_pose(2.8, 7)
            current_state = "SM_WaitForArrival"

        elif current_state == "SM_WaitForArrival":
            if goal_reached:
                say("I arrived at the destination.")
                current_state = "SM_Approach"

        elif current_state == "SM_Approach":
            goal_reached = False
            say("Approaching the table.")
            go_to_goal_pose(3.1, 5.65)
            move_base(0, 0, 1)
            move_head(0, -0.8)
            current_state = "SM_Localize"

        elif current_state == "SM_Localize":
            try:
                say("Looking for the object.")
                x, y, z = find_object(object_name)
                say(f"{object_name.capitalize()} found.")
                transform_target = "shoulders_right_link" if object_name == "pringles" else "shoulders_left_link"
                say("Transforming object coordinates.")
                x, y, z = transform_point(x, y, z, source_frame="kinect_link", target_frame=transform_target)
                if None in (x, y, z):
                    raise ValueError("Transform returned None")
                current_state = "SM_Prepare"
            except Exception as e:
                rospy.logerr(f"Localization error: {e}")
                say("I couldn't find or transform the object.")
                executing_task = False
                new_task = False
                current_state = "SM_Waiting"

        elif current_state == "SM_Prepare":
            say("Preparing arms.")
            move_arm([0,0,0,0,0,0,0], pubRaGoalPose)
            move_arm([0,0,0,0,0,0,0], pubLaGoalPose)
            move_arm([-0.7,0.2,0,1.55,0,1.16,0], pubRaGoalPose)
            move_arm([-0.7,0.2,0,1.55,0,1.16,0], pubLaGoalPose)
            current_state = "SM_Grab"

        elif current_state == "SM_Grab":
            try:
                if any(val is None for val in [x, y, z]):
                    raise ValueError("Object position is None")
                if object_name == "pringles":
                    move_gripper(1, pubLaGoalGrip)
                    q = calculate_inverse_kinematics(x, y, z, -2.24, -1.123, 2, "/manipulation/la_ik_trajectory", [0.84,-0.1,-0.1,-0.68,0,0.41,0])
                    if q and q.points:
                        move_arm_trajectory(q, pubLaGoalTraj)
                        move_gripper(-1, pubLaGoalGrip)
                    else:
                        raise RuntimeError("Left arm IK failed.")
                elif object_name == "drink":
                    move_gripper(1, pubRaGoalGrip)
                    q = calculate_inverse_kinematics(x, y, z, 1.359, 1.496, -1.756, "/manipulation/ra_ik_trajectory", [0.7,0,0,0.2,0.7,1.2,0])
                    if q and q.points:
                        move_arm_trajectory(q, pubRaGoalTraj)
                        move_gripper(-1, pubRaGoalGrip)
                    else:
                        raise RuntimeError("Right arm IK failed.")
                say("Grabbing object.")
                current_state = "SM_Lift"
            except Exception as e:
                rospy.logerr(f"Grabbing failed: {e}")
                say("Grabbing failed.")
                executing_task = False
                new_task = False
                current_state = "SM_Waiting"

        elif current_state == "SM_Lift":
            say("Preparing arm.")
            move_arm([-0.7,0.2,0,1.55,0,1.16,0], pubRaGoalPose)
            move_arm([-0.7,0.2,0,1.55,0,1.16,0], pubLaGoalPose)
            current_state = "SM_GoToLoc"

        elif current_state == "SM_GoToLoc":
            say("Going to the destination.")
            go_to_goal_pose(target_location[0], target_location[1])
            current_state = "SM_INIT"

        loop.sleep()

if __name__ == "__main__":
    main()


    
