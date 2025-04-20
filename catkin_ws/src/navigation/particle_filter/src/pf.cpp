/*
 * MOBILE ROBOTS - UNAM, FI, 2025-2
 * LOCALIZATION BY PARTICLE FILTERS
 *
 * Instructions:
 * Write the code necessary to implement localization by particle filters.
 * Modify only the sections marked with the TODO comment. 
 */
#include <numeric>
#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/GetMap.h"
#include "random_numbers/random_numbers.h"
#include "particle_filter/ray_tracer.h"
#include "tf/transform_listener.h"
#include "geometry_msgs/PoseArray.h"
#include "geometry_msgs/Pose2D.h"
#include "tf/transform_broadcaster.h"
#define DISTANCE_THRESHOLD  0.2
#define ANGLE_THRESHOLD     0.2

#define NOMBRE "Arriaga Vitela Carlos Eduardo"

std::vector<geometry_msgs::Pose2D> get_initial_distribution(int N, float min_x, float max_x, float min_y, float max_y,
                                                             float min_a, float max_a)
{
    random_numbers::RandomNumberGenerator rnd;
    std::vector<geometry_msgs::Pose2D> particles(N);
    for(int i = 0; i < N; i++)
    {	
	particles[i].x = rnd.uniformReal(min_x, max_x);
    	particles[i].y = rnd.uniformReal(min_y, max_y);
    	particles[i].theta = rnd.uniformReal(min_a, max_a);
    }
    
    
    return particles;
}

void move_particles(std::vector<geometry_msgs::Pose2D>& particles, float delta_x, float delta_y, float delta_t, float sigma2)
{
    random_numbers::RandomNumberGenerator rnd;  
    for(int i = 0; i < particles.size(); i++)
    {	
	// * particles[i].x = rnd.uniformReal()
	particles[i].x += delta_x*cos(particles[i].theta) - delta_y*sin(particles[i].theta) + rnd.gaussian(0, sigma2);
    	particles[i].y += delta_x*sin(particles[i].theta) + delta_y*cos(particles[i].theta) + rnd.gaussian(0, sigma2);
    	particles[i].theta += delta_t + rnd.gaussian(0, sigma2);
    }

}

std::vector<sensor_msgs::LaserScan> simulate_particle_scans(std::vector<geometry_msgs::Pose2D>& particles,
                                                             nav_msgs::OccupancyGrid& map, sensor_msgs::LaserScan& sensor_specs)
{
    std::vector<sensor_msgs::LaserScan> simulated_scans(particles.size());
    for(size_t i=0; i < particles.size(); i++)
    {
        geometry_msgs::Pose sensor_pose;
        sensor_pose.position.x    = particles[i].x;
        sensor_pose.position.y    = particles[i].y;
        sensor_pose.orientation.w = cos(particles[i].theta/2);
        sensor_pose.orientation.z = sin(particles[i].theta/2);
        simulated_scans[i] = *particle_filter::simulateRangeScan(map, sensor_pose, sensor_specs);
    }
    return simulated_scans;
}

std::vector<float> calculate_similarities(std::vector<sensor_msgs::LaserScan>& simulated_scans,
                                                   sensor_msgs::LaserScan& real_scan, int downsampling, float sigma2)
{
    std::vector<float> similarities;
    similarities.resize(simulated_scans.size());

for (size_t i=0; i < simulated_scans.size(); i++)
    {
        float delta_sum = 0.0;
        int valid_comparisons = 0;

        for (size_t j = 0; j < simulated_scans[i].ranges.size(); ++j)
        {
            size_t real_index = j * downsampling;
            if (real_index < real_scan.ranges.size())
            {
                float simulated_range = simulated_scans[i].ranges[j];
                float real_range = real_scan.ranges[real_index];

                if (std::isfinite(simulated_range) && std::isfinite(real_range))
                {
                    delta_sum += std::abs(real_range - simulated_range);
                    valid_comparisons++;
                }
            }
        }

        float delta = 0.0;
        if (valid_comparisons > 0)
        {
            delta = delta_sum / valid_comparisons;
        }

        similarities[i] = std::exp(-(delta * delta) / sigma2);
    }

    double sum_similarities = std::accumulate(similarities.begin(), similarities.end(), 0.0);
    if (sum_similarities > 0.0)
    {
        for (float& similarity : similarities)
        {
            similarity /= sum_similarities;
        }
    }
    else
    {
        float uniform_probability = 1.0f / similarities.size();
        for (float& similarity : similarities)
        {
            similarity = uniform_probability;
        }
    }
     
     
     
     
     
     
    return similarities;
}

int random_choice(std::vector<float>& probabilities)
{
    random_numbers::RandomNumberGenerator rnd;

    float x = rnd.uniformReal(0,1);
    int idx = 0;
    while (idx < probabilities.size())
    {
	    if(x < probabilities[idx])
	    {
	        return idx;
	    }
	    else
	    {
     	        x -= probabilities[idx];
	        idx ++;
	    }
    }
    return -1;
}

std::vector<geometry_msgs::Pose2D> resample_particles(std::vector<geometry_msgs::Pose2D>& particles,
                                                      std::vector<float>& probabilities, float sigma2)
{
    random_numbers::RandomNumberGenerator rnd;
    std::vector<geometry_msgs::Pose2D> resampled_particles(particles.size());

    for(size_t i = 0; i < particles.size(); i++)
    {
        int idx = random_choice(probabilities);
        resampled_particles[i] = particles[idx];
        resampled_particles[i].x += rnd.gaussian(0,sigma2);
        resampled_particles[i].y += rnd.gaussian(0,sigma2);
        resampled_particles[i].theta += rnd.gaussian(0,sigma2);
    }

    return resampled_particles;
}



geometry_msgs::Pose2D get_robot_odometry()
{
    tf::TransformListener listener;
    tf::StampedTransform t;
    geometry_msgs::Pose2D pose;
    try{
        listener.waitForTransform("odom", "base_link", ros::Time(0), ros::Duration(1.0));
        listener.lookupTransform("odom", "base_link", ros::Time(0), t);
        pose.x = t.getOrigin().x();
        pose.y = t.getOrigin().y();
        pose.theta = atan2(t.getRotation().z(), t.getRotation().w())*2;
    }
    catch(std::exception &e){
        pose.x = 0;
        pose.y = 0;
        pose.theta = 0;
        std::cout << "Cannot get robot odometry" << std::endl;
    }
    return pose;
}


bool check_displacement(geometry_msgs::Pose2D& delta_pose)
{
    geometry_msgs::Pose2D robot_pose = get_robot_odometry();
    static geometry_msgs::Pose2D last_pose;
    float delta_x = robot_pose.x - last_pose.x;
    float delta_y = robot_pose.y - last_pose.y;
    float delta_a = robot_pose.theta - last_pose.theta;
    if(delta_a >  M_PI) delta_a -= 2*M_PI;
    if(delta_a < -M_PI) delta_a += 2*M_PI;
    if(sqrt(delta_x*delta_x + delta_y*delta_y) > DISTANCE_THRESHOLD || fabs(delta_a) > ANGLE_THRESHOLD)
    {
        last_pose = robot_pose;
        delta_pose.x =  delta_x*cos(robot_pose.theta) + delta_y*sin(robot_pose.theta);
        delta_pose.y = -delta_x*sin(robot_pose.theta) + delta_y*cos(robot_pose.theta);
        delta_pose.theta = delta_a;
        return true;
    }
    return false;
}

geometry_msgs::Pose2D get_robot_pose_estimation(std::vector<geometry_msgs::Pose2D>& particles)
{
    geometry_msgs::Pose2D p;
    float z = 0;
    float w = 0;
    for(size_t i=0; i < particles.size(); i++)
    {
        p.x += particles[i].x;
        p.y += particles[i].y;
        z   += sin(particles[i].theta/2);
        w   += cos(particles[i].theta/2);
    }
    p.x /= particles.size();
    p.y /= particles.size();
    z   /= particles.size();
    w   /= particles.size();
    p.theta = atan2(z, w)*2;
    return p;
}

geometry_msgs::PoseArray get_pose_array(std::vector<geometry_msgs::Pose2D>& particles)
{
    geometry_msgs::PoseArray poses;
    poses.header.frame_id = "map";
    poses.poses.resize(particles.size());
    for(size_t i=0; i < particles.size(); i++)
    {
        poses.poses[i].position.x = particles[i].x;
        poses.poses[i].position.y = particles[i].y;
        poses.poses[i].orientation.z = sin(particles[i].theta/2);
        poses.poses[i].orientation.w = cos(particles[i].theta/2);
    }
    return poses;
}

tf::Transform calculate_and_publish_estimated_pose(std::vector<geometry_msgs::Pose2D>& particles, ros::Publisher* pub)
{
    tf::TransformBroadcaster broadcaster;
    geometry_msgs::Pose2D odom = get_robot_odometry();
    geometry_msgs::Pose2D loc  = get_robot_pose_estimation(particles);
    tf::Transform odom_to_base(tf::Quaternion(0,0,sin(odom.theta/2),cos(odom.theta/2)), tf::Vector3(odom.x,odom.y,0));
    tf::Transform map_to_base(tf::Quaternion(0,0,sin(loc.theta/2),cos(loc.theta/2)), tf::Vector3(loc.x, loc.y, 0));
    tf::Transform map_to_odom = map_to_base*odom_to_base.inverse();
    pub->publish(get_pose_array(particles));
    return map_to_odom;
}

int main(int argc, char** argv)
{
    std::cout << "LOCALIZATION BY PARTICLE FILTERS - " << NOMBRE << std::endl;
    ros::init(argc, argv, "particle_filter");
    ros::NodeHandle n("~");
    ros::Publisher pub_particles = n.advertise<geometry_msgs::PoseArray>("/particle_cloud", 1);
    ros::Rate loop(20);
    tf::TransformBroadcaster broadcaster;
    
    int N_particles;
    float init_min_x;
    float init_min_y;
    float init_min_a;
    float init_max_x;
    float init_max_y;
    float init_max_a;
    int downsampling;
    float sigma2_sensor;
    float sigma2_movement;
    float sigma2_resampling;
    ros::param::param<int>  ("~N", N_particles, 100);
    ros::param::param<float>("~minX", init_min_x, 1.0);
    ros::param::param<float>("~minY", init_min_y, -2.0);
    ros::param::param<float>("~minA", init_min_a, -3.1);
    ros::param::param<float>("~maxX", init_max_x, 10.5);
    ros::param::param<float>("~maxY", init_max_y, 11.0);
    ros::param::param<float>("~maxA", init_max_a, 3.1);
    ros::param::param<int>  ("~ds", downsampling, 10);
    ros::param::param<float>("~s2s", sigma2_sensor, 0.1);
    ros::param::param<float>("~s2m", sigma2_movement, 0.1);
    ros::param::param<float>("~s2r", sigma2_resampling, 0.1);
    
    std::cout << "Initializing particle filter with parameters: " << std::endl;
    std::cout << "N = " << N_particles << "   downsampling = " << downsampling << std::endl;
    std::cout << "Sensor variance: " << sigma2_sensor << "  Movement variance: " << sigma2_movement << "   ";
    std::cout << "Resampling variance: " << sigma2_resampling << std::endl;
    std::cout << "min_x: " << init_min_x << "   min_y: " << init_min_y << "   min_a: " << init_min_a << std::endl;
    std::cout << "max_x: " << init_max_x << "   max_y: " << init_max_y << "   max_a: " << init_max_a << std::endl;
    
    nav_msgs::GetMap srv_get_map;
    nav_msgs::OccupancyGrid static_map;                   //A static map
    ros::service::waitForService("/static_map", ros::Duration(20));
    ros::service::call("/static_map", srv_get_map);
    static_map = srv_get_map.response.map;
    std::cout << "Waiting for laser topic..." << std::endl;
    sensor_msgs::LaserScan real_scan = *ros::topic::waitForMessage<sensor_msgs::LaserScan>("/hardware/scan");
    sensor_msgs::LaserScan sensor_specs = real_scan;
    sensor_specs.angle_increment *= downsampling;
    std::cout << "Real Scan Info: Number of readings: " << real_scan.ranges.size() << std::endl;
    std::cout << "Min angle: " << real_scan.angle_min << std::endl;
    std::cout << "Angle increment: " << real_scan.angle_increment << std::endl;
    geometry_msgs::Pose2D delta_pose;
    check_displacement(delta_pose);

    std::vector<geometry_msgs::Pose2D> particles;        //A set of N particles
    std::vector<sensor_msgs::LaserScan> simulated_scans; //A set of simulated laser readings, one scan per particle
    std::vector<float> similarities;                     //A set of similarities for each particle
    
    particles = get_initial_distribution(N_particles, init_min_x, init_max_x, init_min_y, init_max_y, init_min_a, init_max_a);
    tf::Transform map_to_odom = calculate_and_publish_estimated_pose(particles, &pub_particles);
    while(ros::ok())
    {
        if(check_displacement(delta_pose))
        {
            std::cout << "Displacement detected. Updating pose estimation..." << std::endl;
            real_scan = *ros::topic::waitForMessage<sensor_msgs::LaserScan>("/hardware/scan");
            move_particles(particles, delta_pose.x, delta_pose.y, delta_pose.theta, sigma2_movement);
	    simulated_scans = simulate_particle_scans(particles, static_map, sensor_specs);
	    similarities = calculate_similarities(simulated_scans, real_scan, downsampling, sigma2_sensor);
	    particles = resample_particles(particles, similarities, sigma2_resampling);


            /*
             */
            map_to_odom = calculate_and_publish_estimated_pose(particles, &pub_particles);
        }
        broadcaster.sendTransform(tf::StampedTransform(map_to_odom, ros::Time::now(), "map", "odom"));
        ros::spinOnce();
        loop.sleep();
    }
    return 0;
}
