#include <iostream>
#include <QApplication>
#include "MainWindow.h"
#include "QtRosNode.h"

int main(int argc, char** argv)
{
    std::cout << "INITIALIZING JUSTINA GUI BY MARCOSOFT" << std::endl;
    ros::init(argc, argv, "justina_gui");
    ros::NodeHandle n;
    ros::Rate loop(5);

    std::string la_predefined_file = "";
    std::string ra_predefined_file = "";
    std::string hd_predefined_file = "";
    if(ros::param::has("~la_predefined"))
        ros::param::get("~la_predefined", la_predefined_file);
    if(ros::param::has("~ra_predefined"))
        ros::param::get("~ra_predefined", ra_predefined_file);
    if(ros::param::has("~hd_predefined"))
        ros::param::get("~hd_predefined", hd_predefined_file);

    std::cout << "JustinaGUI.->Waiting for arms initial positions..." << std::endl;
    std_msgs::Float64MultiArray::ConstPtr la_q0, ra_q0;
    int counter = 5;
    while((la_q0 == NULL || ra_q0 == NULL) && ros::ok() && --counter > 0)
    {
        la_q0 = ros::topic::waitForMessage<std_msgs::Float64MultiArray>
            ("/hardware/left_arm/current_pose", ros::Duration(1.0));
        ra_q0 = ros::topic::waitForMessage<std_msgs::Float64MultiArray>
        ("/hardware/right_arm/current_pose", ros::Duration(1.0));
        loop.sleep();
    }
    if(la_q0 == NULL || ra_q0 == NULL)
        std::cout << "JustinaGUI.->Initial position is NULL" << std::endl;
    else
        std::cout << "JustinaGUI.->Arm initial positions received..." << std::endl;

    std::cout << "JustinaGUI.->Trying to load predefined positions files..." << std::endl;
    YamlParser yamlParser;
    yamlParser.loadLaPredefinedPositions(la_predefined_file);
    yamlParser.loadRaPredefinedPositions(ra_predefined_file);
    yamlParser.loadHdPredefinedPositions(hd_predefined_file);
    std::cout << "JustinaGUI.->Predefined position files loaded." << std::endl;

    QtRosNode qtRosNode;
    qtRosNode.setNodeHandle(&n);
    qtRosNode.start();

    QApplication app(argc, argv);
    MainWindow mainWindow;
    mainWindow.setRosNode(&qtRosNode);
    mainWindow.setYamlParser(&yamlParser);
    mainWindow.initArmsGuiElements(la_q0, ra_q0);
        
    mainWindow.show();
    return app.exec();
}
