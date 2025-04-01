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

NAME = "JOSE AUGUSTO ARENAS HERNANDEZ"

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
 
      
    # 1. Asignar los límites inferiores y superiores de color según el objeto solicitado.
   # Validar que la imagen y la nube de puntos no estén vacías
    if img_bgr is None or points is None:
        print("[ERROR] La imagen o la nube de puntos están vacías.")
        return [img_x, img_y, x, y, z]

    # Definir límites de color HSV
    if obj_name == 'pringles':
        lower_color = np.array([25, 50, 50])
        upper_color = np.array([35, 255, 255])
    else:  # 'drink'
        lower_color = np.array([10, 200, 50])
        upper_color = np.array([20, 255, 255])

    # Convertir imagen de BGR a HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Crear máscara binaria con el rango de color
    img_bin = cv2.inRange(img_hsv, lower_color, upper_color)

    # Ver si se detecta alguna región
    non_zero_points = cv2.findNonZero(img_bin)
    if non_zero_points is None:
        print(f"[INFO] No se encontraron píxeles para el objeto '{obj_name}'.")
        return [img_x, img_y, x, y, z]

    # Calcular centroide
    moments = cv2.moments(img_bin)
    if moments['m00'] != 0:
        img_x = int(moments['m10'] / moments['m00'])  # Centroide X en imagen
        img_y = int(moments['m01'] / moments['m00'])  # Centroide Y en imagen
        print(f"[INFO] Centroide en imagen: ({img_x}, {img_y})")

        # Validar que img_x e img_y no estén fuera del rango de la nube de puntos
        h, w, _ = img_bgr.shape
        if 0 <= img_x < w and 0 <= img_y < h:
            x = points[img_y, img_x][0]  # Coordenada X en espacio cartesiano
            y = points[img_y, img_x][1]  # Coordenada Y en espacio cartesiano
            z = points[img_y, img_x][2]  # Coordenada Z en espacio cartesiano
            print(f"[INFO] Centroide en espacio 3D: ({x}, {y}, {z})")
        else:
            print("[ERROR] Centroide fuera de la imagen, no se puede obtener posición 3D.")
    
        
    return [img_x, img_y, x,y,z]

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

