#!/usr/bin/env python3
import rospy
import numpy
from std_msgs.msg import Float64MultiArray, Float64, Bool
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

#THIS NODE RECEIVES A TRAJECTORY FOR THE ARM AND SENDS POSITION ONE BY ONE
#THIS NODE IS INTENDED TO BE USED ONLY IN SIMULATION MODE,
#TRAJECTORY TRACKING FOR REAL AMRS IS PERFORMED IN THE ARMS NODE.

def callback_joint_state(msg):
    global pubLaCurrentPose, pubRaCurrentPose, pubLaCurrentGrip, pubRaCurrentGrip, pubHdCurrentPose
    global hd_current_pose, la_current_pose, ra_current_pose, la_current_gripper, ra_current_gripper
    hd_msg = Float64MultiArray()
    la_msg = Float64MultiArray()
    ra_msg = Float64MultiArray()
    hd_msg.data.append(msg.position[0])
    hd_msg.data.append(msg.position[1])
    for i in range(7):
        la_msg.data.append(msg.position[2+i])
    la_current_gripper = msg.position[9]
    for i in range(7):
        ra_msg.data.append(msg.position[11+i])
    ra_current_gripper = msg.position[18]
    pubHdCurrentPose.publish(hd_msg)
    pubLaCurrentPose.publish(la_msg)
    pubRaCurrentPose.publish(ra_msg)
    hd_current_pose = numpy.asarray(hd_msg.data)
    la_current_pose = numpy.asarray(la_msg.data)
    ra_current_pose = numpy.asarray(ra_msg.data)

        
def callback_hd_q_traj(msg):
    global pubHdPan, pubHdTilt, hd_goal_pose, hd_new_goal
    print("TopicResender.->Startin hd_q_trajectory with " + str(len(msg.points) ) + " points")
    i = 0
    ts = 0.05
    hd_goal_pose = numpy.asarray(msg.points[-1].positions)
    hd_new_goal = True
    for point in msg.points:
        q1, q2= point.positions
        pubHdPan.publish(q1)
        pubHdTilt.publish(q2)
        rospy.sleep(ts)  # Wait ts seconds while moving the arm to the desired position
        i += 1
        #if i >=50:
            #break    
    print("TopicResender.->End of hd_q_trajectory...")

def callback_la_q_traj(msg):
    global pubLaAngle1, pubLaAngle1, pubLaAngle2, pubLaAngle3, pubLaAngle4, pubLaAngle5, pubLaAngle6, pubLaAngle7
    global la_goal_pose, la_new_goal
    print("TopicResender.->Startin la_q_trajectory with " + str(len(msg.points) ) + " points")
    i = 0
    ts = 0.05
    la_goal_pose = numpy.asarray(msg.points[-1].positions)
    la_new_goal = True
    for point in msg.points:
        q1, q2, q3, q4, q5, q6, q7 = point.positions
        pubLaAngle1.publish(q1)
        pubLaAngle2.publish(q2)
        pubLaAngle3.publish(q3)
        pubLaAngle4.publish(q4)
        pubLaAngle5.publish(q5)
        pubLaAngle6.publish(q6)
        pubLaAngle7.publish(q7)
        rospy.sleep(ts)  # Wait ts seconds while moving the arm to the desired position
        i += 1
        #if i >=50:
            #break    
    print("TopicResender.->End of la_q_trajectory...")

def callback_ra_q_traj(msg):
    global pubRaAngle1, pubRaAngle1, pubRaAngle2, pubRaAngle3, pubRaAngle4, pubRaAngle5, pubRaAngle6, pubRaAngle7
    global ra_goal_pose, ra_new_goal
    print("TopicResender.->Startin ra_q_trajectory with " + str(len(msg.points) ) + " points")
    i = 0
    ts = 0.05
    ra_goal_pose = numpy.asarray(msg.points[-1].positions)
    ra_new_goal = True
    for point in msg.points:
        q1, q2, q3, q4, q5, q6, q7 = point.positions
        pubRaAngle1.publish(q1)
        pubRaAngle2.publish(q2)
        pubRaAngle3.publish(q3)
        pubRaAngle4.publish(q4)
        pubRaAngle5.publish(q5)
        pubRaAngle6.publish(q6)
        pubRaAngle7.publish(q7)
        rospy.sleep(ts)  # Wait ts seconds while moving the arm to the desired position
        i += 1
        #if i >=50:
            #break    
    print("TopicResender.->End of ra_q_trajectory...")


def callback_la_goal_pose(msg):
    global pubLaAngle1, pubLaAngle1, pubLaAngle2, pubLaAngle3, pubLaAngle4, pubLaAngle5, pubLaAngle6, pubLaAngle7
    global la_goal_pose, la_new_goal
    if len(msg.data) != 7:
        print("TopicResender.->ERROR!!! left arm goal pose must be a 7-value float64 array. ")
        return
    la_goal_pose = numpy.asarray(msg.data)
    la_new_goal = True
    pubLaAngle1.publish(msg.data[0])
    pubLaAngle2.publish(msg.data[1])
    pubLaAngle3.publish(msg.data[2])
    pubLaAngle4.publish(msg.data[3])
    pubLaAngle5.publish(msg.data[4])
    pubLaAngle6.publish(msg.data[5])
    pubLaAngle7.publish(msg.data[6])

def callback_ra_goal_pose(msg):
    global pubRaAngle1, pubRaAngle1, pubRaAngle2, pubRaAngle3, pubRaAngle4, pubRaAngle5, pubRaAngle6, pubRaAngle7
    global ra_goal_pose, ra_new_goal
    if len(msg.data) != 7:
        print("TopicResender.->ERROR!!! right arm goal pose must be a 7-value float64 array. ")
        return
    ra_goal_pose = msg.data
    ra_new_goal = True
    pubRaAngle1.publish(msg.data[0])
    pubRaAngle2.publish(msg.data[1])
    pubRaAngle3.publish(msg.data[2])
    pubRaAngle4.publish(msg.data[3])
    pubRaAngle5.publish(msg.data[4])
    pubRaAngle6.publish(msg.data[5])
    pubRaAngle7.publish(msg.data[6])

def callback_la_goal_gripper(msg):
    global la_goal_gripper, la_new_goal_gripper
    la_goal_gripper = msg.data
    la_new_goal_gripper = True
    pubLaAngleGr.publish(msg.data)
    pubLaAngleGl.publish(msg.data)

def callback_ra_goal_gripper(msg):
    global ra_goal_gripper, ra_new_goal_gripper
    ra_goal_gripper = msg.data
    ra_new_goal_gripper = True
    pubRaAngleGr.publish(msg.data)
    pubRaAngleGl.publish(msg.data)

def callback_head_goal_pose(msg):
    global hd_goal_pose, hd_new_goal
    if len(msg.data) != 2:
        print("TopicResender.->ERROR!!! head goal pose must be a 2-value float64 array. ")
        return
    hd_goal_pose = numpy.asarray(msg.data)
    hd_new_goal = True
    pubHdPan.publish(msg.data[0])
    pubHdTilt.publish(msg.data[1])    

def main():
    global pubLaAngle1, pubLaAngle1, pubLaAngle2, pubLaAngle3, pubLaAngle4, pubLaAngle5, pubLaAngle6, pubLaAngle7 
    global pubRaAngle1, pubRaAngle1, pubRaAngle2, pubRaAngle3, pubRaAngle4, pubRaAngle5, pubRaAngle6, pubRaAngle7 
    global pubLaAngleGl, pubLaAngleGr, pubRaAngleGl, pubRaAngleGr, pubHdPan, pubHdTilt
    global pubLaCurrentPose, pubRaCurrentPose, pubLaCurrentGrip, pubRaCurrentGrip, pubHdCurrentPose
    global la_current_pose, ra_current_pose, hd_current_pose, la_current_gripper, ra_current_gripper
    global la_goal_pose, ra_goal_pose, hd_goal_pose, la_goal_gripper, ra_goal_gripper
    global la_new_goal, ra_new_goal, hd_new_goal, la_new_goal_gripper, ra_new_goal_gripper

    la_tolerance = 0.2
    ra_tolerance = 0.2
    hd_tolerance = 0.2
    la_gripper_tolerance = 0.2
    ra_gripper_tolerance = 0.2
    print("INITIALIZING TOPIC RESENDER...")
    rospy.init_node("topic_resender")
    
    pubLaGoalReached = rospy.Publisher("/manipulation/left_arm/goal_reached", Bool, queue_size=10);
    pubRaGoalReached = rospy.Publisher("/manipulation/right_arm/goal_reached", Bool, queue_size=10);
    pubHdGoalReached = rospy.Publisher("/manipulation/head/goal_reached", Bool, queue_size=10);
    pubLaGoalGripperReached = rospy.Publisher("/manipulation/left_arm/goal_gripper_reached", Bool, queue_size=10);
    pubRaGoalGripperReached = rospy.Publisher("/manipulation/right_arm/goal_gripper_reached", Bool, queue_size=10);
    pubLaCurrentPose = rospy.Publisher("/hardware/left_arm/current_pose" , Float64MultiArray, queue_size=10)
    pubRaCurrentPose = rospy.Publisher("/hardware/right_arm/current_pose", Float64MultiArray, queue_size=10)
    pubLaCurrentGrip = rospy.Publisher("/hardware/left_arm/current_gripper" , Float64, queue_size=10)
    pubRaCurrentGrip = rospy.Publisher("/hardware/right_arm/current_gripper", Float64, queue_size=10)
    pubHdCurrentPose = rospy.Publisher("/hardware/head/current_pose", Float64MultiArray, queue_size=10);
    pubLaAngle1  = rospy.Publisher("/la_1_controller/command", Float64, queue_size=10)
    pubLaAngle2  = rospy.Publisher("/la_2_controller/command", Float64, queue_size=10)
    pubLaAngle3  = rospy.Publisher("/la_3_controller/command", Float64, queue_size=10)
    pubLaAngle4  = rospy.Publisher("/la_4_controller/command", Float64, queue_size=10)
    pubLaAngle5  = rospy.Publisher("/la_5_controller/command", Float64, queue_size=10)
    pubLaAngle6  = rospy.Publisher("/la_6_controller/command", Float64, queue_size=10)
    pubLaAngle7  = rospy.Publisher("/la_7_controller/command", Float64, queue_size=10)
    pubRaAngle1  = rospy.Publisher("/ra_1_controller/command", Float64, queue_size=10)
    pubRaAngle2  = rospy.Publisher("/ra_2_controller/command", Float64, queue_size=10)
    pubRaAngle3  = rospy.Publisher("/ra_3_controller/command", Float64, queue_size=10)
    pubRaAngle4  = rospy.Publisher("/ra_4_controller/command", Float64, queue_size=10)
    pubRaAngle5  = rospy.Publisher("/ra_5_controller/command", Float64, queue_size=10)
    pubRaAngle6  = rospy.Publisher("/ra_6_controller/command", Float64, queue_size=10)
    pubRaAngle7  = rospy.Publisher("/ra_7_controller/command", Float64, queue_size=10)
    pubLaAngleGl = rospy.Publisher("/la_grip_left_controller/command" , Float64, queue_size=10)
    pubLaAngleGr = rospy.Publisher("/la_grip_right_controller/command", Float64, queue_size=10)
    pubRaAngleGl = rospy.Publisher("/ra_grip_left_controller/command" , Float64, queue_size=10)
    pubRaAngleGr = rospy.Publisher("/ra_grip_right_controller/command", Float64, queue_size=10)
    pubHdPan     = rospy.Publisher("/head_pan_controller/command",  Float64, queue_size=10)
    pubHdTilt    = rospy.Publisher("/head_tilt_controller/command", Float64, queue_size=10)
    rospy.Subscriber("/joint_states", JointState, callback_joint_state)
    rospy.Subscriber("/hardware/left_arm/goal_pose",  Float64MultiArray, callback_la_goal_pose)
    rospy.Subscriber("/hardware/right_arm/goal_pose", Float64MultiArray, callback_ra_goal_pose)
    rospy.Subscriber("/hardware/left_arm/goal_gripper",  Float64, callback_la_goal_gripper)
    rospy.Subscriber("/hardware/right_arm/goal_gripper", Float64, callback_ra_goal_gripper)
    rospy.Subscriber("/hardware/head/goal_pose", Float64MultiArray, callback_head_goal_pose)
    rospy.Subscriber("/manipulation/la_q_trajectory",JointTrajectory, callback_la_q_traj)
    rospy.Subscriber("/manipulation/ra_q_trajectory",JointTrajectory, callback_ra_q_traj)
    rospy.Subscriber("/manipulation/hd_q_trajectory",JointTrajectory, callback_hd_q_traj)
    
    loop = rospy.Rate(50)
    hd_current_pose = numpy.zeros(2)
    la_current_pose = numpy.zeros(7)
    ra_current_pose = numpy.zeros(7)
    la_current_gripper = 0
    ra_current_gripper = 0
    hd_goal_pose = numpy.zeros(2)
    la_goal_pose = numpy.zeros(7)
    ra_goal_pose = numpy.zeros(7)
    la_goal_gripper = 0
    ra_goal_gripper = 0
    hd_new_goal = False
    la_new_goal = False
    ra_new_goal = False
    la_new_goal_gripper = False
    ra_new_goal_gripper = False
    while not rospy.is_shutdown():
        hd_error = numpy.linalg.norm(hd_current_pose - hd_goal_pose)
        la_error = numpy.linalg.norm(la_current_pose - la_goal_pose)
        ra_error = numpy.linalg.norm(ra_current_pose - ra_goal_pose)
        la_error_gripper = abs(la_goal_gripper - la_current_gripper)
        ra_error_gripper = abs(ra_goal_gripper - ra_current_gripper)
        if hd_new_goal and hd_error < hd_tolerance:
            hd_new_goal = False
            pubHdGoalReached.publish(True)
        if la_new_goal and la_error < la_tolerance:
            la_new_goal = False
            pubLaGoalReached.publish(True)
        if ra_new_goal and ra_error < ra_tolerance:
            ra_new_goal = False
            pubRaGoalReached.publish(True)
        if la_new_goal_gripper and la_error_gripper < la_gripper_tolerance:
            la_new_goal_gripper = False
            pubLaGoalGripperReached.publish(True)
        if ra_new_goal_gripper and ra_error_gripper < ra_gripper_tolerance:
            ra_new_goal_gripper = False
            pubRaGoalGripperReached.publish(True)
        loop.sleep()


if __name__ == '__main__':
    main()
    
