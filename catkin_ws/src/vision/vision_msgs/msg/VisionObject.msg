std_msgs/Header header
string id                                    #name of identifying the object (milk, orange juice, etc) or color (green,blue, etc)
string category                              #object type (drink, snack, etc)
float32 confidence                           #value in [0,1] indicating the probability of a correct identification
                                             #in the case of color objects indiates the grasping priority
sensor_msgs/Image image			     #image in rgb of object;
sensor_msgs/Image obj_mask		      #binary image of object
sensor_msgs/PointCloud2 point_cloud          #Point cloud (probably, colored) of the object
geometry_msgs/Vector3 size                   #Size in meters: size in x, y and z
geometry_msgs/Pose pose                      #Centroid and orientation
geometry_msgs/Vector3[] bounding_box         #Two vectors indicating the bounding box
geometry_msgs/Vector3[] bounding_polygon     #N vectors 
int32 x					     #top left x
int32 y					     #top left y
int32 width				     #top widht
int32 height				     #top height
bool graspable                               #graspable by the gripper
std_msgs/ColorRGBA color_rgba	             #Mean object's color
float32 alfa				     #angle of the largest dimension of the object with respect to the surface
string object_state			     #inclination of the collinear axis to the longest dimension of the object
