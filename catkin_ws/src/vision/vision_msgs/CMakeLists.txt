cmake_minimum_required(VERSION 3.0.2)
project(vision_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  message_generation
)

add_message_files(
  FILES
  Keypoint.msg
  HumanCoordinates.msg
  HumanCoordinatesArray.msg
  VisionObject.msg
  VisionObjects.msg
  Person.msg
)


add_service_files(
  FILES
  
  FindLines.srv
  FindPlanes.srv
  HumanPoseEstimatorResult.srv
  RecognizeObjects.srv
  RecognizeObject.srv
  TrainObject.srv
  FindPerson.srv
  PreprocessPointCloud.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

generate_messages(
  DEPENDENCIES
  geometry_msgs   sensor_msgs   std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES vision_msgs
  CATKIN_DEPENDS geometry_msgs roscpp rospy sensor_msgs std_msgs tf
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

