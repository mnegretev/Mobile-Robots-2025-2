#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# FINAL PROJECT – SIMPLE SERVICE ROBOT
#
# Comando de voz esperado (Gramática JSGF):
#   ROBOT TAKE THE <PRINGLES|DRINK> TO THE <TABLE|KITCHEN>
#
# El robot reconoce la orden por voz, describe su progreso con síntesis,
# toma el objeto y lo lleva al destino indicado.
# ──────────────────────────────────────────────────────────────────────

import rospy, tf, time, math
import threading, sys
from std_msgs.msg import Float64MultiArray, Float64, Bool
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from trajectory_msgs.msg import JointTrajectory
from sound_play.msg import SoundRequest
from sensor_msgs.msg import PointCloud2
from hri_msgs.msg import RecognizedSpeech
from vision_msgs.srv import RecognizeObject, RecognizeObjectRequest
from manip_msgs.srv  import (InverseKinematicsPose2Traj,
                             InverseKinematicsPose2TrajRequest,
                             GetPolynomialTrajectory,
                             GetPolynomialTrajectoryRequest)

# ╭──────────────────────────────────────────────╮
# │                Escenario                     │
# ╰──────────────────────────────────────────────╯
NAME          = "Frausto Martinez Juan Carlos"
WP_TABLE      = ( 8.00,  8.50)                  # destino mesa
WP_KITCHEN    = ( 3.22,  9.72)                  # destino cocina
WP_SHELF      = ( 6.50,  8.50)                  # frente al estante
CARRY_POSTURE = [-0.8, -0.2, 1.3, 1.9, 0.0, 1.2, 0.0]  # brazo seguro

# ╭──────────────────────────────────────────────╮
# │               Callbacks TF & voz             │
# ╰──────────────────────────────────────────────╯
def cb_recognized_speech(msg):
    global recognized_speech, new_cmd
    if not executing_task:
        recognized_speech = msg.hypothesis[0]
        new_cmd = True
        rospy.loginfo(f"Recognized: {recognized_speech}")

def cb_goal_reached(msg):
    global goal_reached
    goal_reached = msg.data

# ╭──────────────────────────────────────────────╮
# │                 Funciones I/O                │
# ╰──────────────────────────────────────────────╯
def say(text):
    snd = SoundRequest()
    snd.sound, snd.command, snd.volume       = -3, 1, 1.0
    snd.arg2,  snd.arg                       = "voice_kal_diphone", text
    pubSay.publish(snd)
    rospy.loginfo(f"Saying: {text}")
    rospy.sleep(0.1 * len(text))

def move_head(pan, tilt):
    msg = Float64MultiArray(data=[pan, tilt])
    pubHdGoalPose.publish(msg); rospy.sleep(1.0)

def move_left_arm(*q):
    pubLaGoalPose.publish(Float64MultiArray(data=list(q))); rospy.sleep(2.0)

def move_left_arm_traj(traj):
    pubLaGoalTraj.publish(traj)
    rospy.sleep(0.05 * len(traj.points) + 2)

def move_left_gripper(angle):
    pubLaGoalGrip.publish(Float64(angle)); rospy.sleep(1.0)

def move_base(linear, angular, t):
    cmd = Twist(); cmd.linear.x, cmd.angular.z = linear, angular
    pubCmdVel.publish(cmd); rospy.sleep(t); pubCmdVel.publish(Twist())

def go_to(x, y):
    g = PoseStamped()
    g.pose.orientation.w, g.pose.position.x, g.pose.position.y = 1.0, x, y
    pubGoalPose.publish(g)

# ─── callback opcional de teclado ──────────────────────────────────────
def terminal_listener():
    global recognized_speech, new_cmd
    for line in sys.stdin:                    # bloquea hasta Enter
        if line.strip() and not executing_task:
            recognized_speech = line.strip()
            new_cmd = True
            rospy.loginfo(f"[TTY] {recognized_speech}")

# ╭──────────────────────────────────────────────╮
# │  Servicios de visión y manipulación          │
# ╰──────────────────────────────────────────────╯
def find_object(name:str):
    req = RecognizeObjectRequest()
    req.point_cloud = rospy.wait_for_message(
        "/camera/depth_registered/points", PointCloud2)
    req.name = name
    return srvObjReco(req).recog_object.pose.position

def tf_point(x,y,z, src="realsense_link", dst="shoulder_left_link"):
    try:
        tf_listener.waitForTransform(dst, src, rospy.Time(),
                                     rospy.Duration(4.0))
        p = PointStamped()
        p.header.frame_id, p.header.stamp = src, rospy.Time(0)
        p.point.x, p.point.y, p.point.z   = x,y,z
        p = tf_listener.transformPoint(dst, p)
        return p.point.x, p.point.y, p.point.z
    except (tf.Exception, tf.LookupException, tf.ConnectivityException):
        raise RuntimeError("TF transform failed")

def ik_left(x,y,z,roll=0,pitch=math.pi/2,yaw=0):
    req = InverseKinematicsPose2TrajRequest()
    req.x, req.y, req.z = x,y,z
    req.roll, req.pitch, req.yaw = roll,pitch,yaw
    req.time_step = 0.05
    return srvIKleft(req).articular_trajectory

def poly_traj_left(q_goal, dur=2.0, dt=0.05):
    q_curr = rospy.wait_for_message(
        "/hardware/left_arm/current_pose", Float64MultiArray).data
    req = GetPolynomialTrajectoryRequest(p1=q_curr, p2=q_goal,
                                         duration=dur, time_step=dt)
    return srvPoly(req).trajectory

# ╭──────────────────────────────────────────────╮
# │        Parseo de la frase reconocida         │
# ╰──────────────────────────────────────────────╯
def parse_command(cmd:str):
    cmd = cmd.upper()
    obj = "pringles" if "PRINGLES" in cmd else "drink"
    wp  = WP_TABLE if "TABLE" in cmd else WP_KITCHEN
    wp_name = "table" if wp == WP_TABLE else "kitchen"
    return obj, wp, wp_name

# ╭──────────────────────────────────────────────╮
# │           Máquina de estados (AFSM)          │
# ╰──────────────────────────────────────────────╯
class States:
    INIT, WAIT_CMD, PARSE, APPROACH_OBJ, FIND_OBJ, GRASP, RETRACT, \
    NAV_DEST, WAIT_ARR, RELEASE, FINISH = range(11)

current_state = States.INIT
next_after_arrival = None

# ╭──────────────────────────────────────────────╮
# │               Nodo principal                 │
# ╰──────────────────────────────────────────────╯
if __name__ == "__main__":
    rospy.init_node("final_project")
    tf_listener = tf.TransformListener()

    # Suscriptores
    rospy.Subscriber("/hri/sp_rec/recognized",
                     RecognizedSpeech, cb_recognized_speech)
    rospy.Subscriber("/navigation/goal_reached", Bool, cb_goal_reached)

    # Publicadores
    pubGoalPose   = rospy.Publisher("/move_base_simple/goal",
                                    PoseStamped, queue_size=10)
    pubCmdVel     = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    pubSay        = rospy.Publisher("/hri/speech_generator",
                                    SoundRequest, queue_size=1)
    pubLaGoalPose = rospy.Publisher("/hardware/left_arm/goal_pose",
                                    Float64MultiArray, queue_size=10)
    pubHdGoalPose = rospy.Publisher("/hardware/head/goal_pose",
                                    Float64MultiArray, queue_size=10)
    pubLaGoalGrip = rospy.Publisher("/hardware/left_arm/goal_gripper",
                                    Float64, queue_size=10)
    pubLaGoalTraj = rospy.Publisher("/manipulation/la_q_trajectory",
                                    JointTrajectory, queue_size=10)

    # Servicios
    rospy.loginfo("Waiting for services…")
    rospy.wait_for_service("/manipulation/la_ik_trajectory")
    rospy.wait_for_service("/manipulation/polynomial_trajectory")
    rospy.wait_for_service("/vision/obj_reco/detect_and_recognize_object")
    srvIKleft = rospy.ServiceProxy("/manipulation/la_ik_trajectory",
                                   InverseKinematicsPose2Traj)
    srvPoly   = rospy.ServiceProxy("/manipulation/polynomial_trajectory",
                                   GetPolynomialTrajectory)
    srvObjReco= rospy.ServiceProxy("/vision/obj_reco/detect_and_recognize_object",
                                   RecognizeObject)
    rospy.loginfo("Services ready.")

    # Variables globales de flujo
    new_cmd, executing_task, goal_reached = False, False, False
    recognized_speech = ""

    ENABLE_TTY = rospy.get_param("~enable_terminal", False)

    if ENABLE_TTY:
        t = threading.Thread(target=terminal_listener, daemon=True)
        t.start()
        rospy.loginfo("Terminal input ENABLED – type your commands here.")
    else:
        rospy.loginfo("Terminal input DISABLED – use voice commands.")

    # Presentación
    rospy.loginfo("FINAL PROJECT – " + NAME)
    say("Ready")

    rate = rospy.Rate(10)

    # ══════════════════════════════════════════════════════════════
    #                       BUCLE PRINCIPAL
    # ══════════════════════════════════════════════════════════════
    while not rospy.is_shutdown():

        #  INIT ----------------------------------------------------
        if current_state == States.INIT:
            say("Hello. I'm ready to execute a command.")
            move_head(0,-0.4); move_head(0,0.4); move_head(0,0)
            move_left_arm(0,0,0,0,0,0,0)
            current_state = States.WAIT_CMD

        #  WAIT_CMD -----------------------------------------------
        elif current_state == States.WAIT_CMD:
            if new_cmd:
                new_cmd = False
                executing_task = True
                current_state  = States.PARSE

        #  PARSE ---------------------------------------------------
        elif current_state == States.PARSE:
            target_obj, target_wp, target_wp_name = parse_command(recognized_speech)
            say(f"I will take the {target_obj} to the {target_wp_name}.")
            current_state = States.APPROACH_OBJ

        #  APPROACH_OBJ -------------------------------------------
        elif current_state == States.APPROACH_OBJ:
            say("Approaching the objects.")
            go_to(*WP_SHELF)
            goal_reached = False
            next_after_arrival = States.FIND_OBJ
            current_state = States.WAIT_ARR

        #  FIND_OBJ ------------------------------------------------
        elif current_state == States.FIND_OBJ:
            say(f"Searching for {target_obj}.")
            try:
                pos = find_object(target_obj)
                x,y,z = tf_point(pos.x, pos.y, pos.z)
                traj  = ik_left(x,y,z)
                current_state = States.GRASP
            except Exception as e:
                rospy.logerr(e)
                say("I can't see the object.")
                executing_task  = False
                current_state   = States.WAIT_CMD

        #  GRASP ---------------------------------------------------
        elif current_state == States.GRASP:
            say("Grasping.")
            move_left_arm_traj(traj)
            move_left_gripper(1.0)
            say("Object grasped.")
            current_state = States.RETRACT

        #  RETRACT -------------------------------------------------
        elif current_state == States.RETRACT:
            traj_carry = poly_traj_left(CARRY_POSTURE)
            move_left_arm_traj(traj_carry)
            current_state = States.NAV_DEST

        #  NAV_DEST ------------------------------------------------
        elif current_state == States.NAV_DEST:
            say(f"Navigating to the {target_wp_name}.")
            go_to(*target_wp)
            goal_reached       = False
            next_after_arrival = States.RELEASE
            current_state      = States.WAIT_ARR

        #  WAIT_ARR -----------------------------------------------
        elif current_state == States.WAIT_ARR:
            if goal_reached:
                current_state = next_after_arrival

        #  RELEASE -------------------------------------------------
        elif current_state == States.RELEASE:
            say("Releasing the object.")
            move_left_gripper(0.0)
            move_left_arm(0,0,0,0,0,0,0)
            say("Task completed.")
            current_state = States.FINISH

        #  FINISH --------------------------------------------------
        elif current_state == States.FINISH:
            executing_task, recognized_speech = False, ""
            current_state = States.WAIT_CMD

        rate.sleep()

if __name__ == '__main__':
    main()
