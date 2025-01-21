#include "QtRosNode.h"

QtRosNode::QtRosNode()
{
    this->gui_closed = false;
    publishing_cmd_vel = false;
    cmd_vel.linear.x = 0;
    cmd_vel.linear.x = 0;
    cmd_vel.linear.x = 0;
    cmd_vel.angular.x = 0;
    cmd_vel.angular.y = 0;
    cmd_vel.angular.z = 0;
    la_current_q.resize(7);
    ra_current_q.resize(7);
    la_voltage = 0;
    // la_voltage_bar = 0;
    ra_voltage = 0;
    //ra_voltage_bar = 0;
    la_current_cartesian.resize(6);
    ra_current_cartesian.resize(6);
    /* QString danger = QProgressBar::chunk
        {
            background-color: rgb(115, 210, 22);
            width: 20px;
        };
    QString safe= QProgressBar::chunk
        {
            background-color: rgb(204, 0, 0);
            width: 20px;
         }*/
    spr_recognized = "";
}
QtRosNode::~QtRosNode()
{
}

void QtRosNode::run()
{    
    ros::Rate loop(30);
    pubCmdVel     = n->advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    pubGoalLoc    = n->advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1);
    cltKnownLoc   = n->serviceClient<planning_msgs::GetLocation>("/planning/get_known_location");
    
    pubTorso      = n->advertise<std_msgs::Float64>("/hardware/torso/goal_pose", 1);
    pubLaGoalQ    = n->advertise<std_msgs::Float64MultiArray>("/hardware/left_arm/goal_pose", 1);
    pubRaGoalQ    = n->advertise<std_msgs::Float64MultiArray>("/hardware/right_arm/goal_pose", 1);
    pubLaGoalTraj = n->advertise<trajectory_msgs::JointTrajectory>("/manipulation/la_q_trajectory",1);
    pubRaGoalTraj = n->advertise<trajectory_msgs::JointTrajectory>("/manipulation/ra_q_trajectory",1);
    pubHdGoalQ    = n->advertise<std_msgs::Float64MultiArray>("/hardware/head/goal_pose", 1);
    pubLaGoalGrip = n->advertise<std_msgs::Float64>("/hardware/left_arm/goal_gripper", 1);
    pubRaGoalGrip = n->advertise<std_msgs::Float64>("/hardware/right_arm/goal_gripper", 1);
    subLaCurrentQ = n->subscribe("/hardware/left_arm/current_pose" , 1, &QtRosNode::callback_la_current_q, this);
    subLaVoltage  = n->subscribe("/hardware/left_arm_voltage",1, &QtRosNode::callback_la_voltage,this);
    subRaCurrentQ = n->subscribe("/hardware/right_arm/current_pose", 1, &QtRosNode::callback_ra_current_q, this);
    subRaVoltage  = n->subscribe("/hardware/right_arm_voltage",1, &QtRosNode::callback_ra_voltage,this);
    cltLaIKPose2Traj      =n->serviceClient<manip_msgs::InverseKinematicsPose2Traj>("/manipulation/la_ik_trajectory");
    cltRaIKPose2Traj      =n->serviceClient<manip_msgs::InverseKinematicsPose2Traj>("/manipulation/ra_ik_trajectory");
    cltLaIKPose2Pose      =n->serviceClient<manip_msgs::InverseKinematicsPose2Pose>("/manipulation/la_ik_pose");
    cltRaIKPose2Pose      =n->serviceClient<manip_msgs::InverseKinematicsPose2Pose>("/manipulation/ra_ik_pose");
    cltLaForwardKinematics=n->serviceClient<manip_msgs::ForwardKinematics>("/manipulation/la_forward_kinematics");
    cltRaForwardKinematics=n->serviceClient<manip_msgs::ForwardKinematics>("/manipulation/ra_forward_kinematics");
    cltGetPolynomialTraj  =n->serviceClient<manip_msgs::GetPolynomialTrajectory>("/manipulation/polynomial_trajectory");

    pubSpeechGen       = n->advertise<sound_play::SoundRequest>("/hri/speech_generator", 1);
    pubFakeSpeechRecog = n->advertise<hri_msgs::RecognizedSpeech>("/hri/sp_rec/recognized", 1);
    subRecogSpeech     = n->subscribe("/hri/sp_rec/recognized",1, &QtRosNode::callback_recognized_speech, this);
        
    cltFindLines           = n->serviceClient<vision_msgs::FindLines>           ("/vision/line_finder/find_table_edge");
    cltFindHoriPlanes      = n->serviceClient<vision_msgs::FindPlanes>          ("/vision/line_finder/find_horizontal_plane_ransac");
    cltTrainObject         = n->serviceClient<vision_msgs::TrainObject>         ("/vision/obj_reco/detect_and_train_object");
    cltRecogObjects        = n->serviceClient<vision_msgs::RecognizeObjects>    ("/vision/obj_reco/detect_and_recognize_objects");
    cltRecogObject         = n->serviceClient<vision_msgs::RecognizeObject >    ("/vision/obj_reco/detect_and_recognize_object");
    cltGetPointsAbovePlane = n->serviceClient<vision_msgs::PreprocessPointCloud>("/vision/get_points_above_plane");
    pubHumanPoseEnable     = n->advertise<std_msgs::Bool>("/vision/human_pose/enable", 1);
    pubTakeObject          = n->advertise<std_msgs::String>("/plannning/simple_task/take_object", 1);

    pubLegFinderEnable     = n->advertise<std_msgs::Bool>("/hri/leg_finder/enable", 1);
    pubFollowHumanEnable   = n->advertise<std_msgs::Bool>("/hri/human_following/enable", 1);
    
    int pub_zero_counter = 5;
    while(ros::ok() && !this->gui_closed)
    {
        if(publishing_cmd_vel)
        {
            pubCmdVel.publish(cmd_vel);
            pub_zero_counter = 5;
        }
        else if(--pub_zero_counter > 0)
        {
            if(pub_zero_counter <= 0)
                pub_zero_counter = 0;
            pubCmdVel.publish(cmd_vel);
        }
        ros::spinOnce();
        emit updateGraphics();
        loop.sleep();
    }
    emit onRosNodeFinished();
}

void QtRosNode::setNodeHandle(ros::NodeHandle* nh)
{
    this->n = nh;
}

void QtRosNode::publish_cmd_vel(float linear_frontal, float linear_lateral, float angular)
{
    cmd_vel.linear.x = linear_frontal;
    cmd_vel.linear.y = linear_lateral;
    cmd_vel.angular.z = angular;
    pubCmdVel.publish(cmd_vel);
}

void QtRosNode::publish_goal_location(float x, float y, float a)
{
    geometry_msgs::PoseStamped msg;
    msg.pose.position.x = x;
    msg.pose.position.y = y;
    msg.pose.orientation.w = cos(a/2);
    msg.pose.orientation.z = sin(a/2);
    pubGoalLoc.publish(msg);
}

bool QtRosNode::call_known_location(std::string location, float& x, float& y, float& a)
{
    planning_msgs::GetLocation srv;
    srv.request.name = location;
    if(!cltKnownLoc.call(srv))
        return false;
    x = srv.response.location.pose.position.x;
    y = srv.response.location.pose.position.y;
    a = atan2(srv.response.location.pose.orientation.z, srv.response.location.pose.orientation.w)*2;
    return true;
}

void QtRosNode::start_publishing_cmd_vel(float linear_frontal, float linear_lateral, float angular)
{
    cmd_vel.linear.x = linear_frontal;
    cmd_vel.linear.y = linear_lateral;
    cmd_vel.angular.z = angular;
    publishing_cmd_vel = true;
}

void QtRosNode::stop_publishing_cmd_vel()
{
    cmd_vel.linear.x = 0;
    cmd_vel.linear.y = 0;
    cmd_vel.angular.z = 0;
    publishing_cmd_vel = false;
}

void QtRosNode::get_robot_pose(float& robot_x, float& robot_y, float& robot_a)
{
    tf::StampedTransform t;
    tf::Quaternion q;
    tf_listener.waitForTransform("map", "base_link", ros::Time::now(), ros::Duration(0.5));
    tf_listener.lookupTransform("map", "base_link", ros::Time(0), t);
    robot_x = t.getOrigin().x();
    robot_y = t.getOrigin().y();
    q = t.getRotation();
    robot_a = atan2(q.z(), q.w())*2;
}

void QtRosNode::publish_torso_position(float tr)
{
    std_msgs::Float64 msg;
    msg.data = tr;
    pubTorso.publish(msg);
}

void QtRosNode::publish_la_goal_angles(float a1, float a2, float a3, float a4, float a5, float a6, float a7)
{
    std_msgs::Float64MultiArray msg;
    msg.data.resize(7);
    msg.data[0] = a1;
    msg.data[1] = a2;
    msg.data[2] = a3;
    msg.data[3] = a4;
    msg.data[4] = a5;
    msg.data[5] = a6;
    msg.data[6] = a7;
    pubLaGoalQ.publish(msg);
}

void QtRosNode::publish_ra_goal_angles(float a1, float a2, float a3, float a4, float a5, float a6, float a7)
{
    std_msgs::Float64MultiArray msg;
    msg.data.resize(7);
    msg.data[0] = a1;
    msg.data[1] = a2;
    msg.data[2] = a3;
    msg.data[3] = a4;
    msg.data[4] = a5;
    msg.data[5] = a6;
    msg.data[6] = a7;
    pubRaGoalQ.publish(msg);
}

void QtRosNode::publish_la_goal_trajectory(trajectory_msgs::JointTrajectory Q)
{
    pubLaGoalTraj.publish(Q);
}

void QtRosNode::publish_ra_goal_trajectory(trajectory_msgs::JointTrajectory Q)
{
    pubRaGoalTraj.publish(Q);
}

void QtRosNode::publish_la_grip_angles(float a)
{
    std_msgs::Float64 msg;
    msg.data = a;
    pubLaGoalGrip.publish(msg);
}

void QtRosNode::publish_ra_grip_angles(float a)
{
    std_msgs::Float64 msg;
    msg.data = a;
    pubRaGoalGrip.publish(msg);
}

void QtRosNode::publish_head_angles(double pan, double tilt)
{
    std_msgs::Float64MultiArray msg;
    msg.data.resize(2);
    msg.data[0] = pan;
    msg.data[1] = tilt;
    pubHdGoalQ.publish(msg);
}

void QtRosNode::callback_la_current_q(const std_msgs::Float64MultiArray::ConstPtr& msg)
{
    la_current_q = msg->data;
    call_la_forward_kinematics(la_current_q, la_current_cartesian);
}

void QtRosNode::callback_la_voltage(const std_msgs::Float64::ConstPtr& msg)
{
    la_voltage = msg->data;
}
void QtRosNode::callback_ra_current_q(const std_msgs::Float64MultiArray::ConstPtr& msg)
{
    ra_current_q = msg->data;
    call_ra_forward_kinematics(ra_current_q, ra_current_cartesian);
}

void QtRosNode::callback_ra_voltage(const std_msgs::Float64::ConstPtr& msg)
{
    ra_voltage = msg->data;
}

bool QtRosNode::call_la_ik_trajectory(std::vector<double>& cartesian, trajectory_msgs::JointTrajectory& trajectory)
{
    manip_msgs::InverseKinematicsPose2Traj srv;
    srv.request.x = cartesian[0];
    srv.request.y = cartesian[1];
    srv.request.z = cartesian[2];
    srv.request.roll  = cartesian[3];
    srv.request.pitch = cartesian[4];
    srv.request.yaw   = cartesian[5];
    if(!cltLaIKPose2Traj.call(srv))
        return false;
    trajectory = srv.response.articular_trajectory;
    return true;
}

bool QtRosNode::call_ra_ik_trajectory(std::vector<double>& cartesian, trajectory_msgs::JointTrajectory& trajectory)
{
    manip_msgs::InverseKinematicsPose2Traj srv;
    srv.request.x = cartesian[0];
    srv.request.y = cartesian[1];
    srv.request.z = cartesian[2];
    srv.request.roll  = cartesian[3];
    srv.request.pitch = cartesian[4];
    srv.request.yaw   = cartesian[5];
    if(!cltRaIKPose2Traj.call(srv))
        return false;
    trajectory = srv.response.articular_trajectory;
    return true;
}

bool QtRosNode::call_la_ik_pose(std::vector<double>& cartesian, std::vector<double>& articular)
{
    manip_msgs::InverseKinematicsPose2Pose srv;
    srv.request.x = cartesian[0];
    srv.request.y = cartesian[1];
    srv.request.z = cartesian[2];
    srv.request.roll  = cartesian[3];
    srv.request.pitch = cartesian[4];
    srv.request.yaw   = cartesian[5];
    if(!cltLaIKPose2Pose.call(srv))
        return false;
    articular = srv.response.q;
    return true;
}

bool QtRosNode::call_ra_ik_pose(std::vector<double>& cartesian, std::vector<double>& articular)
{
    manip_msgs::InverseKinematicsPose2Pose srv;
    srv.request.x = cartesian[0];
    srv.request.y = cartesian[1];
    srv.request.z = cartesian[2];
    srv.request.roll  = cartesian[3];
    srv.request.pitch = cartesian[4];
    srv.request.yaw   = cartesian[5];
    if(!cltRaIKPose2Pose.call(srv))
        return false;
    articular = srv.response.q;
    return true;
}

bool QtRosNode::call_la_forward_kinematics(std::vector<double>& articular, std::vector<double>& cartesian)
{
    cartesian.resize(6);
    for(int i=0; i<cartesian.size(); i++) cartesian[i] = 0;
    manip_msgs::ForwardKinematics srv;
    srv.request.q = articular;
    if(!cltLaForwardKinematics.call(srv))
        return false;
    cartesian[0] = srv.response.x;
    cartesian[1] = srv.response.y;
    cartesian[2] = srv.response.z;
    cartesian[3] = srv.response.roll;
    cartesian[4] = srv.response.pitch;
    cartesian[5] = srv.response.yaw;
    return true;

}

bool QtRosNode::call_ra_forward_kinematics(std::vector<double>& articular, std::vector<double>& cartesian)
{
    cartesian.resize(6);
    for(int i=0; i<cartesian.size(); i++) cartesian[i] = 0;
    manip_msgs::ForwardKinematics srv;
    srv.request.q = articular;
    if(!cltRaForwardKinematics.call(srv))
        return false;
    cartesian[0] = srv.response.x;
    cartesian[1] = srv.response.y;
    cartesian[2] = srv.response.z;
    cartesian[3] = srv.response.roll;
    cartesian[4] = srv.response.pitch;
    cartesian[5] = srv.response.yaw;
    return true;
}

bool QtRosNode::call_get_polynomial_traj(std::vector<double>& p1, std::vector<double>& p2, trajectory_msgs::JointTrajectory& trajectory)
{
    manip_msgs::GetPolynomialTrajectory srv;
    srv.request.p1 = p1;
    srv.request.p2 = p2;
    srv.request.time_step  = 0.05;
    double max_delta = -1;
    for(size_t i=0; i < p1.size(); i++)
        if(fabs(p1[i] - p2[i]) > max_delta)
            max_delta = fabs(p1[i] - p2[i]);
    srv.request.duration = max_delta / 0.7 + 0.5;
    if(!cltGetPolynomialTraj.call(srv))
        return false;
    trajectory = srv.response.trajectory;
    return true;
}

void QtRosNode::say(std::string text_to_say)
{
    sound_play::SoundRequest msg;
    msg.sound   = -3;
    msg.volume  = 1.0;
    msg.command = 1;
    msg.arg     = text_to_say;
    msg.arg2    = "voice_cmu_us_slt_arctic_hts";
    pubSpeechGen.publish(msg);
}

void QtRosNode::publish_fake_speech_recog(std::string text_to_say)
{
    hri_msgs::RecognizedSpeech msg;
    msg.hypothesis.push_back(text_to_say);
    msg.confidences.push_back(0.9);
    pubFakeSpeechRecog.publish(msg);
}

void QtRosNode::callback_recognized_speech(const hri_msgs::RecognizedSpeech::ConstPtr& msg)
{
    if(msg->hypothesis.size() < 1)
        return;
    spr_recognized = msg->hypothesis[0];
}

bool QtRosNode::call_find_lines()
{
    vision_msgs::FindLines srv;
    boost::shared_ptr<sensor_msgs::PointCloud2 const> ptr;
    ptr = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/camera/depth_registered/points", ros::Duration(1.0));
    if(ptr==NULL)
    {
        std::cout << "JustinaGUI.->Cannot get point cloud before calling find lines service..." << std::endl;
        return false;
    }
    srv.request.point_cloud = *ptr;
    bool success = cltFindLines.call(srv);
    if(!success) std::cout << "JustinaGUI.->Cannot find any lines :'(" << std::endl;
    return success;
}

bool QtRosNode::call_find_horizontal_planes()
{
    vision_msgs::FindPlanes srv;
    boost::shared_ptr<sensor_msgs::PointCloud2 const> ptr;
    ptr = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/camera/depth_registered/points", ros::Duration(1.0));
    if(ptr==NULL)
    {
        std::cout << "JustinaGUI.->Cannot get point cloud before calling find planes service..." << std::endl;
        return false;
    }
    srv.request.point_cloud = *ptr;
    bool success = cltFindHoriPlanes.call(srv);
    if(!success) std::cout << "JustinaGUI.->Cannot find any planes :'(" << std::endl;
    return success;
}

bool QtRosNode::call_train_object(std::string name)
{
    vision_msgs::TrainObject srv;
    boost::shared_ptr<sensor_msgs::PointCloud2 const> ptr;
    ptr = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/camera/depth_registered/points", ros::Duration(1.0));
    if(ptr==NULL)
    {
        std::cout << "JustinaGUI.->Cannot get point cloud before calling train object service..." << std::endl;
        return false;
    }
    srv.request.point_cloud = *ptr;
    srv.request.name = name;
    return cltTrainObject.call(srv);
}

bool QtRosNode::call_recognize_objects()
{
    vision_msgs::RecognizeObjects srv;
    boost::shared_ptr<sensor_msgs::PointCloud2 const> ptrCloud;
    boost::shared_ptr<sensor_msgs::Image const> ptrImg;
    ptrCloud = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/camera/depth_registered/points", ros::Duration(1.0));
    //ptrImg   = ros::topic::waitForMessage<sensor_msgs::Image>("/camera/depth_registered/rgb/image_raw", ros::Duration(1.0));
    if(ptrCloud==NULL)
        std::cout << "JustinaGUI.->WARNING!!!! Cannot get point cloud before calling recognize object service..." << std::endl;
    else
        srv.request.point_cloud = *ptrCloud;
    // if(ptrImg == NULL)
    //     std::cout << "JustinaGUI.->WARNING!!!! Cannot get image before calling recognize object service..." << std::endl;
    // else
    //     srv.request.image = *ptrImg;
    
    bool success = cltRecogObjects.call(srv);
    if(success)
    {
        std::cout << "JustinaGUI.->Found objects:" << std::endl;
        for(int i=0; i < srv.response.recog_objects.size(); i++)
            std::cout << "JustinaGUI.->" << srv.response.recog_objects[i].id << "  " << srv.response.recog_objects[i].pose << std::endl;
    }else
        std::cout << "JustinaGUI.->Cannot find any objects ':(" << std::endl;
    return success;
}

bool QtRosNode::call_recognize_object(std::string name)
{
    vision_msgs::RecognizeObject srv;
    boost::shared_ptr<sensor_msgs::PointCloud2 const> ptr;
    ptr = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/camera/depth_registered/points", ros::Duration(1.0));
    if(ptr==NULL)
    {
        std::cout << "JustinaGUI.->Cannot get point cloud before calling train object service..." << std::endl;
        return false;
    }
    srv.request.point_cloud = *ptr;
    srv.request.name = name;
    bool success = cltRecogObject.call(srv);
    if(success)
        std::cout << "JustinaGUI.->Found " << srv.response.recog_object.id << " at " << srv.response.recog_object.pose << std::endl;
    else
        std::cout << "JustinaGUI.->Cannot find " << srv.request.name << std::endl;
    return success;
}


void QtRosNode::call_take_object(std::string name)
{
    std_msgs::String msg;
    msg.data = name;

    std::cout << "JustinaGUI.->Take object**************************************************" << std::endl;
    
    /*
    vision_msgs::RecognizeObject srv;
    boost::shared_ptr<sensor_msgs::PointCloud2 const> ptr;
    ptr = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/camera/depth_registered/points", ros::Duration(1.0));
    if(ptr==NULL)
    {
        std::cout << "JustinaGUI.->Cannot get point cloud before calling train object service..." << std::endl;
        return false;
    }
    
    srv.request.point_cloud = *ptr;
    srv.request.name = name;
    */
    pubTakeObject.publish(msg);
    msg.data = " ";
    pubTakeObject.publish(msg);
}

bool QtRosNode::call_get_points_above_plane()
{
    vision_msgs::PreprocessPointCloud srv;
    boost::shared_ptr<sensor_msgs::PointCloud2 const> ptr;
    ptr = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/camera/depth_registered/points", ros::Duration(1.0));
    if(ptr==NULL)
    {
        std::cout << "JustinaGUI.->Cannot get point cloud before calling preprocess object service..." << std::endl;
        return false;
    }
    srv.request.input_cloud = *ptr;
    return cltGetPointsAbovePlane.call(srv);
}

void QtRosNode::publish_enable_human_pose_detection(bool enable)
{
    std_msgs::Bool msg;
    msg.data = enable;
    pubHumanPoseEnable.publish(msg);
}

void QtRosNode::publish_enable_human_following(bool enable)
{
    std_msgs::Bool msg;
    msg.data = enable;
    pubLegFinderEnable.publish(msg);
    pubFollowHumanEnable.publish(msg);
}
