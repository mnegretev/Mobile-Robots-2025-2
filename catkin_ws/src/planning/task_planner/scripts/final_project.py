def main():
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
    pubLaGoalPose = rospy.Publisher("/hardware/left_arm/goal_pose" , Float64MultiArray, queue_size=10)
    pubRaGoalPose = rospy.Publisher("/hardware/right_arm/goal_pose", Float64MultiArray, queue_size=10)
    pubHdGoalPose = rospy.Publisher("/hardware/head/goal_pose"     , Float64MultiArray, queue_size=10)
    pubLaGoalGrip = rospy.Publisher("/hardware/left_arm/goal_gripper" , Float64, queue_size=10)
    pubRaGoalGrip = rospy.Publisher("/hardware/right_arm/goal_gripper", Float64, queue_size=10)
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
    object_name = ""
    target_location = []
    say("Ready")
    x, y, z = 0, 0, 0

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
            say("Going to the table.")
            go_to_goal_pose(target_location[0], target_location[1])
            current_state = "SM_WaitForArrival"

        elif current_state == "SM_WaitForArrival":
            if goal_reached:
                say("I arrived at the table.")
                current_state = "SM_RotateInPlace"

        elif current_state == "SM_RotateInPlace":
            say("Searching for object.")
            move_head(0, -0.5)
            current_state = "SM_Localize"

        elif current_state == "SM_Localize":
            try:
                x, y, z = find_object(object_name)
                say(f"{object_name} found.")
                if object_name == "pringles":
                    x, y, z = transform_point(x, y, z, "realsense_link", "shoulders_right_link")
                else:
                    x, y, z = transform_point(x, y, z, "realsense_link", "shoulders_left_link")
                current_state = "SM_Prepare"
            except:
                say("Object not found.")
                current_state = "SM_Waiting"
                new_task = False
                executing_task = False

        elif current_state == "SM_Prepare":
            say("Preparing to grab.")
            move_left_arm(0, 0, 0, 0, 0, 0, 0)  # Ajusta si quieres una pose inicial distinta
            current_state = "SM_Grab"

        elif current_state == "SM_Grab":
            traj = calculate_inverse_kinematics_left(x, y, z, 0, 0, 0)
            move_left_arm_with_trajectory(traj)
            move_left_gripper(-1.0)
            say("Object grabbed.")
            current_state = "SM_Lift"

        elif current_state == "SM_Lift":
            move_left_arm(0, 0.3, 0, 0, 0, 0, 0)  # Levanta el objeto un poco
            current_state = "SM_GoToLoc"

        elif current_state == "SM_GoToLoc":
            say("Delivering object.")
            go_to_goal_pose(target_location[0], target_location[1])
            current_state = "SM_WaitAtDestination"

        elif current_state == "SM_WaitAtDestination":
            if goal_reached:
                say("Arrived at destination.")
                move_left_gripper(1.0)  # Soltar objeto
                move_left_arm(0, 0, 0, 0, 0, 0, 0)  # Volver a reposo
                say("Task completed.")
                executing_task = False
                new_task = False
                current_state = "SM_Waiting"

        loop.sleep()

