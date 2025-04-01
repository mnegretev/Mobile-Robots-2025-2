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

NAME = "MURILLO SANTOS JAVIER EDUARDO"

def segment_by_color(img_bgr, points, obj_name):
    global img_hsv, img_bin, img_filtered
    img_x, img_y, x, y, z = 0, 0, 0, 0, 0

    # Asignar límites HSV según el objeto
    if obj_name == "pringles":
        lower = numpy.array([25, 50, 50])
        upper = numpy.array([35, 255, 255])
    else:
        lower = numpy.array([10, 200, 50])
        upper = numpy.array([20, 255, 255])

    # Convertir imagen de BGR a HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Crear máscara binaria según los límites de color
    img_bin = cv2.inRange(img_hsv, lower, upper)

    # Aplicar filtro de mediana para eliminar puntos aislados
    img_bin = cv2.medianBlur(img_bin, 5)

    # Calcular el centroide de la máscara (coordenadas de imagen)
    M = cv2.moments(img_bin)
    if M["m00"] > 0:
        img_x = int(M["m10"] / M["m00"])
        img_y = int(M["m01"] / M["m00"])
        
        # Obtener coordenadas cartesianas desde la nube de puntos (índice correcto: [y, x])
        try:
            if not numpy.isnan(points[img_y, img_x][0]):
                x = points[img_y, img_x][0]
                y = points[img_y, img_x][1]
                z = points[img_y, img_x][2]
        except:
            pass


    return [img_x, img_y, x, y, z]


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

