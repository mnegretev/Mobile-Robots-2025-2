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

import numpy
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

NAME = "FULL NAME"
listener = None
ROTATION_TARGET = math.pi*3/2

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
    destiny = "table" if "TABLE" in cmd else "kitchen"
    loc = [6.0, 6.0] if "TABLE" in cmd else [8.0, 0.0]
    return obj, loc, destiny

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
    print(f"Goal: ({goal_x}, {goal_y}) published")

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
    listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(10.0))
    obj_p = PointStamped()
    obj_p.header.frame_id = source_frame
    obj_p.header.stamp = rospy.Time(0)
    obj_p.point.x, obj_p.point.y, obj_p.point.z = x,y,z
    obj_p = listener.transformPoint(target_frame, obj_p)
    return [obj_p.point.x, obj_p.point.y, obj_p.point.z]

def get_robot_pose():
    global listener
    try:
        ([x, y, z], [qx,qy,qz,qw]) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
        return numpy.asarray([x, y]), 2*math.atan2(qz, qw)
    except Exception as e:
        print(f"Error: {e}")
        return numpy.asarray([0,0]),0
    
class State:
    """
    Represents a state in a finite state machine (FSM).
    Attributes:
        name (str): The name of the state.
        say (str, optional): A message to be spoken or displayed when entering the state.
        execute (callable, optional): A function to execute when the state is entered.
        transition (callable, optional): A function that determines when to transition to the next state.
        next (State or tuple of State, optional): The next state(s) to transition to.
    Methods:
        __say__(): Prints the state name and calls the say function if a message is provided.
        __execute__(): Executes the associated function if provided.
        __transition__(): Determines and returns the next state based on the transition function and next attribute.
        run(): Executes the state's say, execute, and transition logic in order, and returns the next state.
    """
    def __init__(self, name, say=None, execute=None, transition = None, next = None):
        self.name = name
        self.say = say
        self.execute = execute
        self.transition = transition
        self.next = next
    
    def __say__(self):
        if self.say:
            print(f"State: {self.name}")
            if type(self.say) is str:
                say(self.say)
            else:
                say(self.say())

    
    def __execute__(self):
        if self.execute:
            self.execute()

    def __transition__(self):
        if self.transition:
            if type(self.next) is tuple:
                if self.transition():
                    return self.next[0]
                else:
                    return self.next[1]
            else:
                while not self.transition():
                    time.sleep(0.5)
                return self.next
        else:
            return self.next
        
    def run(self):
        self.__say__()
        self.__execute__()
        return self.__transition__()
    
class FSM:
    """
    Finite State Machine (FSM) class for managing state transitions.
    Attributes:
        current_state (State): The current active state of the FSM.
        states (dict): A dictionary mapping state names to State instances.
    Args:
        initial_state (State): The initial state to start the FSM.
        states (dict, optional): A dictionary of all possible states. Defaults to None.
    Methods:
        run():
            Executes the FSM loop, calling the `run` method of the current state.
            Transitions to the next state as returned by the current state's `run` method.
            Prints the name of the next state during each transition.
            Stops execution if the next state is None or not found in the states dictionary.
    """
    def __init__(self, initial_state: State, states: dict= None):
        self.current_state = initial_state
        self.states = states if states else {}
    
    def run(self):
        while True:
            next_state = self.current_state.run()
            print(f"Transitioning to: {next_state}")
            if next_state is None:
                break
            self.current_state = self.states.get(next_state)
            if self.current_state is None:
                print(f"State '{next_state}' not found.")
                break
    

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

    recognized_speech = ""
    new_task = False
    executing_task = False
    target_adquired = False
    destiny = ""

    targets_coordinates = [3.2, 6.0]
    goal_coordinates = [0.0, 0.0]  # Placeholder for goal coordinates
    t_pos = None
    left_arm = False
    
    def idle_execute():
        global recognized_speech, new_task, executing_task, goal_coordinates, target_object, target_adquired, destiny
        while not new_task:
            if recognized_speech:
                new_task = True
        target_object, goal_coordinates, destiny = parse_command(recognized_speech.upper())
        print(f"Recognized speech: {recognized_speech}")
        print(f"Target object: {target_object}, Goal coordinates: {goal_coordinates}")
        executing_task = True
        target_adquired = False

    def navigate_execute():
        global goal_reached, target_adquired
        goal_reached = False
        goal = []
        if target_adquired:
            goal = goal_coordinates
        else:
            goal = targets_coordinates
        print(f"Navigating to {goal}")
        go_to_goal_pose(goal[0], goal[1])

    def end_execute():
        global executing_task, new_task
        executing_task = False
        new_task = False
        print("Task completed successfully")

    def navigate_say():
        global target_object, target_adquired, destiny
        return f"Navigating to {target_object if not target_adquired else destiny}"
    
    def detect_execute():
        global t_pos, left_arm
        say(f"Rotating base to position")
        _, rotation = get_robot_pose()
        while abs(rotation - ROTATION_TARGET) > 0.1:
            # Normalize angles to the range [-pi, pi]
            rotation_normalized = (rotation + math.pi) % (2 * math.pi) - math.pi
            target_normalized = (ROTATION_TARGET + math.pi) % (2 * math.pi) - math.pi
            angle_diff = target_normalized - rotation_normalized

            # Ensure the shortest rotation direction
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Determine rotation direction and move base
            angular_speed = 0.75 if angle_diff > 0 else -0.75
            move_base(0.0, angular_speed, 0.1)

            _, rotation = get_robot_pose()
            print(f"Rotation: {rotation} (Normalized: {rotation_normalized}), Target: {ROTATION_TARGET} (Normalized: {target_normalized}), Angle Diff: {angle_diff}")
        say("Turning head down")
        move_head(0, -0.5)
        say("Searching for target object")
        global target_object
        t_pos = find_object(target_object)
        left_arm = True
        if t_pos[1] > 0:
            left_arm = False
        say(f"Using {'left' if left_arm else 'right'} arm")
        t_pos = transform_point(t_pos[0], t_pos[1], t_pos[2], target_frame="shoulders_left_link" if left_arm else "shoulders_right_link")
        
    def prepare_execute():
        global t_pos, left_arm
        say("Preparing arm")
        if left_arm:
            move_left_arm(-0.69, 0.2, 0.0, 1.55, 0.0, 1.16, 0.0)
            q = calculate_inverse_kinematics_left(t_pos[0], t_pos[1], t_pos[2], 2.736, -1.221, -2.897)
            q = get_la_polynomial_trajectory(q, 5, 0.025)
            move_left_arm_with_trajectory(q)
        else:
            move_right_arm(-0.69, 0.2, 0.0, 1.55, 0.0, 1.16, 0.0)
            q = calculate_inverse_kinematics_right(t_pos[0], t_pos[1], t_pos[2], 2.736, -1.221, -2.897)
            q = get_ra_polynomial_trajectory(q, 5, 0.025)
            move_right_arm_with_trajectory(q)
        say("Arm is ready")

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
        transition=lambda: goal_reached,
        next="SelectAction"
    )

    select_action = State(
        "SelectAction",
        say="",
        execute=lambda: None,
        transition=lambda: not target_adquired,
        next = ("DetectTarget", "PrepareArm")
    )

    prepare_arm = State(
        "PrepareArm",
        say="Preparing for target adquisition",
        execute=prepare_execute,
        next="OpenGripper"
    )

    open_gripper = State(
        "OpenGripper",
        say="Opening gripper",
        execute=lambda: print("Gripper is open"),
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
        execute=lambda: print("Robot position is ready, arm is ready"),
        next="GrabTarget"
    )
    
    def grab_target_execute():
        global target_adquired
        print("Target object grabbed")
        target_adquired = True

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
        "PrepareArm": prepare_arm,
        "OpenGripper": open_gripper,
        "EndState": end_state,
        "DetectTarget": detect_target,
        "PrepareRobot": prepare_robot,
        "GrabTarget": grab_target
    }

    fsm = FSM(initial_state=idle, states=states)

    #
    # FINAL PROJECT 
    #
    say("Ready")
    while not rospy.is_shutdown():
        fsm.run()
        
        loop.sleep()

if __name__ == '__main__':
    main()

