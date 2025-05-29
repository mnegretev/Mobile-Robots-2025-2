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

from manip_msgs.srv import InverseKinematicsPose2Traj, InverseKinematicsPose2TrajRequest

   
    
    
def main():
    rospy.set_param("/tf2_buffer_server/ignore_repeated_tfs", True)
    global new_task, recognized_speech, executing_task, goal_reached
    global pubLaGoalPose, pubRaGoalPose, pubHdGoalPose, pubLaGoalGrip, pubRaGoalGrip
    global pubLaGoalTraj, pubRaGoalTraj, pubGoalPose, pubCmdVel, pubSay
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
    

    rospy.init_node("test_ik_client")
rospy.wait_for_service("/manipulation/la_ik_trajectory")

try:
    clt = rospy.ServiceProxy("/manipulation/la_ik_trajectory", InverseKinematicsPose2Traj)
    req = InverseKinematicsPose2TrajRequest()
    req.x = 0.3
    req.y = 0.0
    req.z = 0.2
    req.roll = 0
    req.pitch = -1.57
    req.yaw = 0
    req.duration = 2.0
    req.time_step = 0.05
    req.initial_guess = [0.0]*7
    resp = clt(req)
    print("Respuesta:", resp)
except rospy.ServiceException as e:
    print("Error al llamar servicio:", e)          

            
        loop.sleep()

if __name__ == '__main__':
    main()

