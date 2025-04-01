#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-2
# COLOR SEGMENTATION USING HSV
#
# Instructions:
# Write the code necessary to detect and localize the 'pringles'
# or 'drink' using only a hsv color segmentation.
# MODIFY ONLY THE SECTIONS MARKED WITH THE 'TODO' COMMENT
#


import numpy
import cv2
import ros_numpy
import rospy
import math
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped, Point
from vision_msgs.srv import RecognizeObject, RecognizeObjectResponse

NAME = "LUJAN PEREZ CARLOS EDUARDO"

def segment_by_color(img_bgr, points, obj_name):
    global img_hsv, img_bin, img_filtered
    img_x, img_y, x,y,z = 0,0,0,0,0
    #
    # TODO:
    # - Assign lower and upper color limits according to the requested object:
    #   If obj_name == 'pringles': [25, 50, 50] - [35, 255, 255]
    #   otherwise                : [10,200, 50] - [20, 255, 255]
    # - Change color space from RGB to HSV.
    #   Check online documentation for cv2.cvtColor function
    # - Determine the pixels whose color is in the selected color range.
    #   Check online documentation for cv2.inRange
    # - Calculate the centroid of all pixels in the given color range (ball position).
    #   Check online documentation for cv2.findNonZero and cv2.mean
    # - Calculate the centroid of the segmented region in the cartesian space
    #   using the point cloud 'points'. Use numpy array notation to process the point cloud data.
    #   Example: 'points[240,320][1]' gets the 'y' value of the point corresponding to
    #   the pixel in the center of the image.
    #
    lowerLim = (25, 50, 50) if obj_name == "pringles" else (15, 150, 150)
    upperLim = (35, 255, 255) if obj_name == "pringles" else (20, 255, 200)
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerLim, upperLim)
    test1 = mask
    k_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_size + 1, 2 * k_size + 1))
    eroded = cv2.erode(mask, kernel)
    dilated = cv2.dilate(eroded, kernel)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_mask = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    print(contours)
    for i in range(len(contours)):
        cv2.drawContours(color_mask, contours, i, (100 * i, 255 - 50 * i, 0), -1)
    test2 = color_mask
    idxs = cv2.findNonZero(mask)
    mean = numpy.zeros(3)
    counter = 0
    for [[c,r]] in idxs:
        p = points[r,c]
        p = numpy.array([p[0], p[1], p[2]])
        if(numpy.isnan(numpy.min(p))):
            continue
        mean += p
        counter += 1
    mean /= counter
    img_center = cv2.mean(idxs)
    print("XYZ: ", mean)
    print("IMG: ", img_center)
    return [img_center[0], img_center[1], mean[0], mean[1], mean[2]]
    #return [img_x, img_y, x,y,1]

def callback_find_object(req):
    global pub_point, img_bgr
    print("Trying to find object: " + req.name)
    arr = ros_numpy.point_cloud2.pointcloud2_to_array(req.point_cloud)
    rgb_arr = arr['rgb'].copy()
    rgb_arr.dtype = numpy.uint32
    r = numpy.asarray(((rgb_arr >> 16) & 255), dtype='uint8')
    g = numpy.asarray(((rgb_arr >>  8) & 255), dtype='uint8')
    b = numpy.asarray(((rgb_arr      ) & 255), dtype='uint8')
    img_bgr = cv2.merge((b,g,r))
    [r, c, x, y, z] = segment_by_color(img_bgr, arr, req.name)
    resp = RecognizeObjectResponse()
    resp.recog_object.header.frame_id = 'kinect_link'
    resp.recog_object.header.stamp    = rospy.Time.now()
    resp.recog_object.pose.position.x = x
    resp.recog_object.pose.position.y = y
    resp.recog_object.pose.position.z = z
    pub_point.publish(PointStamped(header=resp.recog_object.header, point=Point(x=x, y=y, z=z)))
    cv2.circle(img_bgr, (int(r), int(c)), 20, [0, 255, 0], thickness=3)
    return resp

def main():
    global pub_point, img_bgr, img_hsv, img_bin, img_filtered
    print("COLOR SEGMENTATION - " + NAME)
    rospy.init_node("color_segmentation")
    rospy.Service("/vision/obj_reco/detect_and_recognize_object", RecognizeObject, callback_find_object)
    pub_point = rospy.Publisher('/detected_object', PointStamped, queue_size=10)
    img_bgr = numpy.zeros((480, 640, 3), numpy.uint8)
    img_hsv = numpy.zeros((480, 640, 3), numpy.uint8)
    img_bin = numpy.zeros((480, 640, 3), numpy.uint8)
    img_filtered = numpy.zeros((480, 640, 3), numpy.uint8)
    loop = rospy.Rate(10)
    while not rospy.is_shutdown():
        cv2.imshow("BGR", img_bgr)
        cv2.imshow("HSV", img_hsv)
        cv2.imshow("Binary", img_bin)
        cv2.imshow("Filtered", img_filtered)
        cv2.waitKey(1)
        loop.sleep()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

