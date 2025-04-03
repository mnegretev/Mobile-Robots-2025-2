/*
 * MOBILE ROBOTS - UNAM, FI, 2025-2
 * LOCALIZATION BY PARTICLE FILTERS
 *
 * Instructions:
 * Write the code necessary to implement localization by particle filters.
 * Modify only the sections marked with the TODO comment. 
 */

#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/GetMap.h"
#include "random_numbers/random_numbers.h"
#include "particle_filter/ray_tracer.h"
#include "tf/transform_listener.h"
#include "geometry_msgs/PoseArray.h"
#include "geometry_msgs/Pose2D.h"
#include <numeric>
#include "tf/transform_broadcaster.h"
#define DISTANCE_THRESHOLD  0.2
#define ANGLE_THRESHOLD     0.2

#define NOMBRE "ARENAS HERNÀNDEZ JOSÈ AUGUSTO"

std::vector<geometry_msgs::Pose2D> get_initial_distribution(int N, float min_x, float max_x, float min_y, float max_y,
                                                            float min_a, float max_a)
{
    random_numbers::RandomNumberGenerator rnd;
    std::vector<geometry_msgs::Pose2D> particles(N);

    /*
     * TODO:
     * Generate a set of N particles (each particle represented by a Pose2D message)
     * with positions uniformly distributed within bounding box given by min_x, ..., max_a.
     * To generate uniformly distributed random numbers, you can use the funcion rnd.uniformReal(min, max)
     */

    for (int i = 0; i < N; ++i) {
        particles[i].x = rnd.uniformReal(min_x, max_x);
        particles[i].y = rnd.uniformReal(min_y, max_y);
        particles[i].theta = rnd.uniformReal(min_a, max_a);
    }

    /*
     */
    return particles;
}

void move_particles(std::vector<geometry_msgs::Pose2D>& particles, float delta_x, float delta_y, float delta_t, float sigma2)
{
    random_numbers::RandomNumberGenerator rnd;
    /*
     * TODO:
     * Move each particle a displacement given by delta_x, delta_y and delta_t.
     * Displacement is given w.r.t. particles's frame, i.e., to calculate the new position for
     * each particle you need to rotate delta_x and delta_y, on Z axis, an angle theta_i, where theta_i
     * is the orientation of the i-th particle.
     * Add gaussian noise to each new position. Use sigma2 as variance.
     * You can use the function rnd.gaussian(mean, variance)
     */
     
      for (size_t i = 0; i < particles.size(); ++i)
    {
        float rotated_x = delta_x * cos(particles[i].theta) - delta_y * sin(particles[i].theta);
        float rotated_y = delta_x * sin(particles[i].theta) + delta_y * cos(particles[i].theta);
      
        particles[i].x += rotated_x + rnd.gaussian(0.0, sigma2);
        particles[i].y += rotated_y + rnd.gaussian(0.0, sigma2);
        particles[i].theta += delta_t + rnd.gaussian(0.0, sigma2);
      
        if (particles[i].theta > M_PI)
            particles[i].theta -= 2 * M_PI;
        if (particles[i].theta < -M_PI)
            particles[i].theta += 2 * M_PI;
    }

}

std::vector<sensor_msgs::LaserScan> simulate_particle_scans(std::vector<geometry_msgs::Pose2D>& particles,
                                                             nav_msgs::OccupancyGrid& map, sensor_msgs::LaserScan& sensor_specs)
{
    /*
     * TODO:
     * Review the code to simulate a laser scan for each particle given the set of particles and a static map. 
     */
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
    /*
     * TODO:
     * For each particle, calculate the similarity between its simulated scan and the real scan.
     * Normalize all similarities (the sum of all values must always be 1.0)
     * Store results in 'similarities'.
     * IMPORTANT NOTE 1. The real sensor scans are DOWNSAMPLED. That is, only 1 out of 'downsampling' scans is considered, i.e.,
     * For example, if downsampling=10, then, if real sensor has 500 ranges, simulated scans will only have 50 ranges
     * When comparing readings, for each reading in the simulated scan, you should skip 'downsampling' readings
     * in the real sensor.
     * IMPORTANT NOTE 2. Both, simulated an real scans, can have infinite distances. Thus, when comparing readings,
     * ensure both simulated and real ranges are finite values. 
     */
    
    /*
     */
    for (size_t i = 0; i < simulated_scans.size(); ++i)
    {
        const auto& simulated_ranges = simulated_scans[i].ranges;
        const auto& real_ranges = real_scan.ranges;
        float sum_delta = 0.0;
        int valid_count = 0;

        for (size_t j = 0; j < simulated_ranges.size(); ++j)
        {
            size_t real_index = j * downsampling;
            if (real_index >= real_ranges.size())
                break;

            if (std::isfinite(simulated_ranges[j]) && std::isfinite(real_ranges[real_index]))
            {
                sum_delta += std::abs(real_ranges[real_index] - simulated_ranges[j]);
                ++valid_count;
            }
        }

        if (valid_count > 0)
        {
            float delta = sum_delta / valid_count;
            similarities[i] = std::exp(-(delta * delta) / sigma2);
        }
        else
        {
            similarities[i] = 0.0;
        }
    }

    float sum_similarities = std::accumulate(similarities.begin(), similarities.end(), 0.0f);
    if (sum_similarities > 0.0)
    {
        for (auto& similarity : similarities)
        {
            similarity /= sum_similarities;
        }
    }
     
    return similarities;
}

int random_choice(std::vector<float>& probabilities)
{
    random_numbers::RandomNumberGenerator rnd;
    /*
     * TODO:
     *
     * Write an algorithm to choice an integer in the range [0, N-1], with N, the length of 'probabilities'.
     * Probability of picking an integer 'i' is given by the corresponding probabilities[i] value.
     * Return the chosen integer. 
     */
    
    /*
     */
    float cumulative_sum = 0.0;
    float random_value = rnd.uniformReal(0.0, 1.0);
    for (size_t i = 0; i < probabilities.size(); ++i)
    {
        cumulative_sum += probabilities[i];
        if (random_value <= cumulative_sum)
        {
            return i;
        }
    }

    return -1;
}

std::vector<geometry_msgs::Pose2D> resample_particles(std::vector<geometry_msgs::Pose2D>& particles,
                                                      std::vector<float>& probabilities, float sigma2)
{
    random_numbers::RandomNumberGenerator rnd;
    std::vector<geometry_msgs::Pose2D> resampled_particles(particles.size());
    /*
     * TODO:
     * Sample, with replacement, N particles from the set 'particles'.
     * The probability of the i-th particle to be resampled is given by probabilities[i].
     * Use the random_choice function to pick a particle with the correct probability.
     * Add gaussian noise to each sampled particle (add noise to x,y and theta). Use sigma2 as noise variance.
     */
    
    /*
     */
    for (size_t i = 0; i < particles.size(); ++i)
    {
        int selected_index = random_choice(probabilities);
        resampled_particles[i] = particles[selected_index];
        resampled_particles[i].x += rnd.gaussian(0.0, sigma2);
        resampled_particles[i].y += rnd.gaussian(0.0, sigma2);
        resampled_particles[i].theta += rnd.gaussian(0.0, sigma2);
        if (resampled_particles[i].theta > M_PI)
            resampled_particles[i].theta -= 2 * M_PI;
        if (resampled_particles[i].theta < -M_PI)
            resampled_particles[i].theta += 2 * M_PI;
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
            /*
             * TODO: Review the functions to:
             * Move all particles a displacement given by delta_pose (Pose2D message) by calling the move_particles function.
             * Get the set of simulated scans for each particles. Use the simulate_particle_scans function.
             * Get the set of similarities by calling the calculate_similarities function
             * Resample particles by calling the resample_particles function
             */
            

            move_particles(particles, delta_pose.x, delta_pose.y, delta_pose.theta, sigma2_movement);
            simulated_scans = simulate_particle_scans(particles, static_map, sensor_specs);
            similarities = calculate_similarities(simulated_scans, real_scan, downsampling, sigma2_sensor);
            particles = resample_particles(particles, similarities, sigma2_resampling);
            map_to_odom = calculate_and_publish_estimated_pose(particles, &pub_particles);
        }
        broadcaster.sendTransform(tf::StampedTransform(map_to_odom, ros::Time::now(), "map", "odom"));
        ros::spinOnce();
        loop.sleep();
    }
    return 0;
}
