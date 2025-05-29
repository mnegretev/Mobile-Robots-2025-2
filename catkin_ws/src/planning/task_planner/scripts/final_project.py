#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# FINAL PROJECT - SIMPLE SERVICE ROBOT
#

import numpy
import rospy # type: ignore
import tf # type: ignore
import math
import time
from std_msgs.msg import String, Float64MultiArray, Float64, Bool  # type: ignore
from nav_msgs.msg import Path # type: ignore
from nav_msgs.srv import GetPlan, GetPlanRequest # type: ignore
from sensor_msgs.msg import PointCloud2 # type: ignore
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point, PointStamped # type: ignore
from trajectory_msgs.msg import JointTrajectory # type: ignore
from sound_play.msg import SoundRequest # type: ignore
from vision_msgs.srv import * # type: ignore
from manip_msgs.srv import * # type: ignore
from hri_msgs.msg import * # type: ignore
from fsm import FSM, State

# =========================
# Global Variables Section
# =========================

class RobotState:
    def __init__(self):
        self.NAME = "FULL NAME"
        self.listener = None
        self.recognized_speech = ""
        self.new_task = False
        self.executing_task = False
        self.goal_reached = False
        self.target_adquired = False
        self.destiny = ""
        self.target_object = ""
        self.pubLaGoalPose = None
        self.pubRaGoalPose = None
        self.pubHdGoalPose = None
        self.pubLaGoalGrip = None
        self.pubRaGoalGrip = None
        self.pubLaGoalTraj = None
        self.pubRaGoalTraj = None
        self.pubGoalPose = None
        self.pubCmdVel = None
        self.pubSay = None
        self.pub_point = None
        self.targets_coordinates = {
            'drink': [3.1, 6.0],
            'pringles': [3.3, 6.0],
            'table': [3.2, 9.0],
            'kitchen': [6.6, -1.0]
        }
        self.rotation_target = {
            'initial': math.pi*3/2,
            'drink': math.pi*3/2 - 0.3,
            'pringles': math.pi*3/2 + 0.3,
            'table': math.pi*3/2,
            'kitchen': math.pi*3/2
        }
        self.goal_coordinates = [0.0, 0.0]
        self.t_pos = None

state: RobotState = None
# =========================

def callback_recognized_speech(msg):
    if state.executing_task:
        return
    state.new_task = True
    state.recognized_speech = msg.hypothesis[0]
    print("New command recognized: " + state.recognized_speech)

def callback_goal_reached(msg):
    state.goal_reached = msg.data
    print("Received goal reached: " + str(state.goal_reached))

def parse_command(cmd):
    obj = "pringles" if "PRINGLES" in cmd else "drink"
    destiny = "table" if "TABLE" in cmd else "kitchen"
    loc = state.targets_coordinates[destiny]
    return obj, loc, destiny

def move_arm(*q):
    pub = state.pubLaGoalPose
    msg = Float64MultiArray()
    msg.data.extend(q)
    pub.publish(msg)
    time.sleep(2.0)

def move_arm_with_trajectory(Q):
    pub = state.pubLaGoalTraj
    pub.publish(Q)
    time.sleep(0.05*len(Q.points) + 2)

def move_gripper(q):
    pub = state.pubLaGoalGrip
    pub.publish(q)
    time.sleep(1.0)

def move_head(pan, tilt):
    msg = Float64MultiArray()
    msg.data.extend([pan, tilt])
    state.pubHdGoalPose.publish(msg)
    time.sleep(1.0)

def move_base(linear, angular, t):
    cmd = Twist()
    cmd.linear.x = linear
    cmd.angular.z = angular
    state.pubCmdVel.publish(cmd)
    time.sleep(t)
    state.pubCmdVel.publish(Twist())

def go_to_goal_pose(goal_x, goal_y):
    goal_pose = PoseStamped()
    goal_pose.pose.orientation.w = 1.0
    goal_pose.pose.position.x = goal_x
    goal_pose.pose.position.y = goal_y
    state.pubGoalPose.publish(goal_pose)
    print(f"Goal: ({goal_x}, {goal_y}) published")

def say(text):
    msg = SoundRequest()
    msg.sound   = -3
    msg.command = 1
    msg.volume  = 1.0
    msg.arg2    = "voice_kal_diphone"
    msg.arg = text
    print("Saying: " + text)
    state.pubSay.publish(msg)
    time.sleep(len(text)*0.1)

def calculate_inverse_kinematics(x, y, z, roll, pitch, yaw):
    req_ik = InverseKinematicsPose2TrajRequest() # type: ignore
    req_ik.x, req_ik.y, req_ik.z = x, y, z
    req_ik.roll, req_ik.pitch, req_ik.yaw = roll, pitch, yaw
    req_ik.duration = 0
    req_ik.time_step = 0.05
    req_ik.initial_guess = []
    srv = "/manipulation/la_ik_trajectory"
    clt = rospy.ServiceProxy(srv, InverseKinematicsPose2Traj) # type: ignore
    resp = clt(req_ik)
    return resp.articular_trajectory

def get_polynomial_trajectory(q, duration=2.0, time_step=0.05):
    topic = "/hardware/left_arm/current_pose"
    current_p = rospy.wait_for_message(topic, Float64MultiArray).data
    clt = rospy.ServiceProxy("/manipulation/polynomial_trajectory", GetPolynomialTrajectory) # type: ignore
    req = GetPolynomialTrajectoryRequest() # type: ignore
    req.p1 = current_p
    req.p2 = q
    req.duration = duration
    req.time_step = time_step
    resp = clt(req)
    return resp.trajectory

def find_object(object_name):
    clt_find_object = rospy.ServiceProxy("/vision/obj_reco/detect_and_recognize_object", RecognizeObject) # type: ignore
    req_find_object = RecognizeObjectRequest() # type: ignore
    req_find_object.point_cloud = rospy.wait_for_message("/camera/depth_registered/points", PointCloud2)
    req_find_object.name  = object_name
    resp = clt_find_object(req_find_object)
    x = resp.recog_object.pose.position.x
    y = resp.recog_object.pose.position.y
    z = resp.recog_object.pose.position.z
    state.pub_point.publish(PointStamped(header=resp.recog_object.header, point=Point(x=x, y=y, z=z)))
    return [x, y, z]

def transform_point(x, y, z, source_frame="kinect_link", target_frame="shoulders_left_link"):
    state.listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
    obj_p = PointStamped()
    obj_p.header.frame_id = source_frame
    obj_p.header.stamp = rospy.Time(0)
    obj_p.point.x, obj_p.point.y, obj_p.point.z = x, y, z
    obj_p = state.listener.transformPoint(target_frame, obj_p)
    return [obj_p.point.x, obj_p.point.y, obj_p.point.z]

def ortho_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_robot_pose():
    try:
        ([x, y, z], [qx,qy,qz,qw]) = state.listener.lookupTransform('map', 'base_link', rospy.Time(0))
        return numpy.asarray([x, y]), 2*math.atan2(qz, qw)
    except Exception as e:
        print(f"Error: {e}")
        return numpy.asarray([0,0]),0

def rotate_to_target(target_angle):
    _, rotation = get_robot_pose()
    while abs(rotation - target_angle) > 0.05:
        rotation_normalized = (rotation + math.pi) % (2 * math.pi) - math.pi
        target_normalized = (target_angle + math.pi) % (2 * math.pi) - math.pi
        angle_diff = target_normalized - rotation_normalized
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        angular_speed = 0.75 if angle_diff > 0 else -0.75
        move_base(0.0, angular_speed, 0.05)
        _, rotation = get_robot_pose()

def move_to_position(target_pos, threshold=0.4, speed=0.25):
    pos, _ = get_robot_pose()
    while ortho_distance(pos, target_pos) > threshold:
        move_base(speed, 0.0, 0.05)
        pos, _ = get_robot_pose()

def prepare_arm_for_detection():
    move_arm(-0.49, 0.0, 0.0, 2.15, 0.0, 1.36, 0.0)
    move_gripper(0.3)

def main():
    global state
    state = RobotState()

    print("FINAL PROJECT - " + state.NAME)
    rospy.init_node("final_project")
    rospy.Subscriber('/hri/sp_rec/recognized', RecognizedSpeech, callback_recognized_speech) # type: ignore
    rospy.Subscriber('/navigation/goal_reached', Bool, callback_goal_reached)
    state.pubGoalPose   = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    state.pubCmdVel     = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    state.pubSay        = rospy.Publisher('/hri/speech_generator', SoundRequest, queue_size=1)
    state.pubLaGoalPose = rospy.Publisher("/hardware/left_arm/goal_pose" , Float64MultiArray, queue_size=10)
    state.pubRaGoalPose = rospy.Publisher("/hardware/right_arm/goal_pose", Float64MultiArray, queue_size=10)
    state.pubHdGoalPose = rospy.Publisher("/hardware/head/goal_pose"     , Float64MultiArray, queue_size=10)
    state.pubLaGoalGrip = rospy.Publisher("/hardware/left_arm/goal_gripper" , Float64, queue_size=10)
    state.pubLaGoalTraj = rospy.Publisher("/manipulation/la_q_trajectory", JointTrajectory, queue_size=10)
    state.pub_point = rospy.Publisher('/detected_object', PointStamped, queue_size=10)
    state.listener = tf.TransformListener()
    print("Waiting for services...")
    rospy.wait_for_service('/manipulation/la_ik_pose')
    rospy.wait_for_service('/manipulation/ra_ik_pose')
    rospy.wait_for_service('/vision/obj_reco/detect_and_recognize_object')
    print("Services are now available.")
    loop = rospy.Rate(10)

    

    

    def idle_execute():
        rospy.sleep(0.1)
        while not state.new_task:
            pass
        state.target_object, state.goal_coordinates, state.destiny = parse_command(state.recognized_speech.upper())
        print(f"Recognized speech: {state.recognized_speech}")
        print(f"Target object: {state.target_object}, Goal coordinates: {state.goal_coordinates}")
        state.executing_task = True
        state.target_adquired = False
        move_arm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def navigate_execute():
        state.goal_reached = False
        goal = state.goal_coordinates if state.target_adquired else state.targets_coordinates[state.target_object]
        print(f"Navigating to {goal}")
        go_to_goal_pose(goal[0], goal[1])

    def end_execute():
        state.executing_task = False
        state.new_task = False
        print("Task completed successfully")

    def navigate_say():
        return f"Navigating to {state.target_object if not state.target_adquired else state.destiny}"

    def detect_execute():
        say("Rotating base to position")
        rotate_to_target(state.rotation_target["initial"])

        say("Turning head down")
        move_head(0, -1.0)

        say("Detecting object")
        state.t_pos = find_object(state.target_object)

        say("Preparing arm")
        prepare_arm_for_detection()

        say("Getting closer to target")
        x, y = state.targets_coordinates[state.target_object]
        move_to_position([x, y - 0.73])

        say("Readjusting base rotation")
        rotate_to_target(state.rotation_target[state.target_object])

        say(f"Using left arm to grab {state.target_object}")

    def prepare_robot_execute():
        say("Preparing arm")
        state.t_pos = find_object(state.target_object)
        move_arm(-0.59, 0.0, 0.0, 1.75, 0.0, 0.56, 0.0)
        move_arm(-0.1432, 0.0, 0.0, 1.8418, 0.0, 0.1695, 0.0)
        state.t_pos = transform_point(state.t_pos[0], state.t_pos[1], state.t_pos[2])
        print(f"Detected position: {state.t_pos}")
        say("Calculating inverse kinematics")
        q = calculate_inverse_kinematics(state.t_pos[0] + 0.15, state.t_pos[1], state.t_pos[2], 0.0, -1.473, 0.0)
        move_arm_with_trajectory(q)
        say("Arm is ready")

    def prepare_arm_execute():
        say("Rotating robot")
        rotate_to_target(state.rotation_target[state.destiny])
        say("Moving arm")
        move_arm(1.0, 0.0, 0.0, 1.35, 0.0, 0.0, 0.0)

    def end_execute():
        say("Task completed successfully")
        state.goal_reached = False
        state.target_adquired = False
        state.destiny = ""
        state.target_object = ""
        move_arm(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        move_gripper(0.0)

    idle = State(
        "Idle",
        say="Waiting for commands",
        execute=idle_execute,
        next="Navigate"
    )

    navigate = State(
        "Navigate",
        say=navigate_say,
        execute=navigate_execute,
        transition=lambda: state.goal_reached,
        next="SelectAction"
    )

    select_action = State(
        "SelectAction",
        say="",
        execute=lambda: None,
        transition=lambda: not state.target_adquired,
        next=("DetectTarget", "PrepareArm")
    )

    prepare_arm_state = State(
        "PrepareArm",
        say="Preparing for target deposition",
        execute=prepare_arm_execute,
        next="OpenGripper"
    )

    open_gripper = State(
        "OpenGripper",
        say="Opening gripper",
        execute=lambda: (say("Opening gripper"), move_gripper(0.6)),
        next="EndState"
    )

    end_state = State(
        "EndState",
        say="Task completed",
        execute=end_execute,
        next="Idle"
    )

    detect_target = State(
        "DetectTarget",
        say="Detecting target object",
        execute=detect_execute,
        next="PrepareRobot"
    )

    prepare_robot = State(
        "PrepareRobot",
        say="Preparing robot for target acquisition",
        execute=prepare_robot_execute,
        next="GrabTarget"
    )

    def grab_target_execute():
        print("Target object grabbed")
        move_gripper(-0.2)  # Close gripper
        move_arm(1.0, 0.0, 0.0, 2.15, 0.0, 2.15, 0.0)
        state.target_adquired = True

    grab_target = State(
        "GrabTarget",
        say="Grabbing target object",
        execute=grab_target_execute,
        next="Navigate"
    )

    states = {
        "Idle": idle,
        "Navigate": navigate,
        "SelectAction": select_action,
        "PrepareArm": prepare_arm_state,
        "OpenGripper": open_gripper,
        "EndState": end_state,
        "DetectTarget": detect_target,
        "PrepareRobot": prepare_robot,
        "GrabTarget": grab_target
    }

    fsm = FSM(initial_state=idle, states=states, say=say)

    say("Ready")
    while not rospy.is_shutdown():
        fsm.run()
        loop.sleep()

if __name__ == '__main__':
    main()


