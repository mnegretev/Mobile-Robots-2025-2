#include "MainWindow.h"
#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    
    QIcon icoFwd(":/images/btnUp");
    QIcon icoBwd(":/images/btnDown");
    QIcon icoLeft(":/images/btnLeft");
    QIcon icoRight(":/images/btnRight");
    QIcon icoTurnLeft(":/images/btnTurnLeft");
    QIcon icoTurnRight(":/images/btnTurnRight");
    ui->btnFwd->setIcon(icoFwd);
    ui->btnBwd->setIcon(icoBwd);
    ui->btnTurnLeft->setIcon(icoTurnLeft);
    ui->btnTurnRight->setIcon(icoTurnRight);
    ui->btnLeft->setIcon(icoLeft);
    ui->btnRight->setIcon(icoRight);

    QObject::connect(ui->btnFwd, SIGNAL(pressed()), this, SLOT(btnFwdPressed()));
    QObject::connect(ui->btnFwd, SIGNAL(released()), this, SLOT(btnFwdReleased()));
    QObject::connect(ui->btnBwd, SIGNAL(pressed()), this, SLOT(btnBwdPressed()));
    QObject::connect(ui->btnBwd, SIGNAL(released()), this, SLOT(btnBwdReleased()));
    QObject::connect(ui->btnLeft, SIGNAL(pressed()), this, SLOT(btnLeftPressed()));
    QObject::connect(ui->btnLeft, SIGNAL(released()), this, SLOT(btnLeftReleased()));
    QObject::connect(ui->btnRight, SIGNAL(pressed()), this, SLOT(btnRightPressed()));
    QObject::connect(ui->btnRight, SIGNAL(released()), this, SLOT(btnRightReleased()));
    QObject::connect(ui->btnTurnLeft, SIGNAL(pressed()), this, SLOT(btnTurnLeftPressed()));
    QObject::connect(ui->btnTurnLeft, SIGNAL(released()), this, SLOT(btnTurnLeftReleased()));
    QObject::connect(ui->btnTurnRight, SIGNAL(pressed()), this, SLOT(btnTurnRightPressed()));
    QObject::connect(ui->btnTurnRight, SIGNAL(released()), this, SLOT(btnTurnRightReleased()));
    QObject::connect(ui->btnCmdVel, SIGNAL(pressed()), this, SLOT(btnCmdVelPressed()));
    QObject::connect(ui->btnCmdVel, SIGNAL(released()), this, SLOT(btnCmdVelReleased()));

    QObject::connect(ui->navTxtStartPose, SIGNAL(returnPressed()), this, SLOT(navBtnCalcPath_pressed()));
    QObject::connect(ui->navTxtGoalPose, SIGNAL(returnPressed()), this, SLOT(navBtnCalcPath_pressed()));
    QObject::connect(ui->navBtnCalcPath, SIGNAL(clicked()), this, SLOT(navBtnCalcPath_pressed()));
    QObject::connect(ui->navBtnExecPath, SIGNAL(clicked()), this, SLOT(navBtnExecPath_pressed()));

    QObject::connect(ui->torTxtPos, SIGNAL(valueChanged(double)), this, SLOT(torSbPosValueChanged(double)));

    QObject::connect(ui->laBtnXp    , SIGNAL(clicked()), this, SLOT(laBtnXpPressed()));
    QObject::connect(ui->laBtnXm    , SIGNAL(clicked()), this, SLOT(laBtnXmPressed()));
    QObject::connect(ui->laBtnYp    , SIGNAL(clicked()), this, SLOT(laBtnYpPressed()));
    QObject::connect(ui->laBtnYm    , SIGNAL(clicked()), this, SLOT(laBtnYmPressed()));
    QObject::connect(ui->laBtnZp    , SIGNAL(clicked()), this, SLOT(laBtnZpPressed()));
    QObject::connect(ui->laBtnZm    , SIGNAL(clicked()), this, SLOT(laBtnZmPressed()));
    QObject::connect(ui->laBtnRollp , SIGNAL(clicked()), this, SLOT(laBtnRollpPressed()));
    QObject::connect(ui->laBtnRollm , SIGNAL(clicked()), this, SLOT(laBtnRollmPressed()));
    QObject::connect(ui->laBtnPitchp, SIGNAL(clicked()), this, SLOT(laBtnPitchpPressed()));
    QObject::connect(ui->laBtnPitchm, SIGNAL(clicked()), this, SLOT(laBtnPitchmPressed()));
    QObject::connect(ui->laBtnYawp  , SIGNAL(clicked()), this, SLOT(laBtnYawpPressed()));
    QObject::connect(ui->laBtnYawm  , SIGNAL(clicked()), this, SLOT(laBtnYawmPressed()));
    QObject::connect(ui->laTxtAngles1, SIGNAL(valueChanged(double)), this, SLOT(laSbAnglesValueChanged(double)));
    QObject::connect(ui->laTxtAngles2, SIGNAL(valueChanged(double)), this, SLOT(laSbAnglesValueChanged(double)));
    QObject::connect(ui->laTxtAngles3, SIGNAL(valueChanged(double)), this, SLOT(laSbAnglesValueChanged(double)));
    QObject::connect(ui->laTxtAngles4, SIGNAL(valueChanged(double)), this, SLOT(laSbAnglesValueChanged(double)));
    QObject::connect(ui->laTxtAngles5, SIGNAL(valueChanged(double)), this, SLOT(laSbAnglesValueChanged(double)));
    QObject::connect(ui->laTxtAngles6, SIGNAL(valueChanged(double)), this, SLOT(laSbAnglesValueChanged(double)));
    QObject::connect(ui->laTxtAngles7, SIGNAL(valueChanged(double)), this, SLOT(laSbAnglesValueChanged(double)));
    QObject::connect(ui->laTxtAnglesG, SIGNAL(valueChanged(double)), this, SLOT(laSbGripperValueChanged(double)));
    QObject::connect(ui->laTxtArticularGoal, SIGNAL(returnPressed()), this, SLOT(laTxtArticularGoalReturnPressed()));
    QObject::connect(ui->laTxtCartesianGoal, SIGNAL(returnPressed()), this, SLOT(laTxtCartesianGoalReturnPressed()));

    QObject::connect(ui->raBtnXp    , SIGNAL(clicked()), this, SLOT(raBtnXpPressed()));
    QObject::connect(ui->raBtnXm    , SIGNAL(clicked()), this, SLOT(raBtnXmPressed()));
    QObject::connect(ui->raBtnYp    , SIGNAL(clicked()), this, SLOT(raBtnYpPressed()));
    QObject::connect(ui->raBtnYm    , SIGNAL(clicked()), this, SLOT(raBtnYmPressed()));
    QObject::connect(ui->raBtnZp    , SIGNAL(clicked()), this, SLOT(raBtnZpPressed()));
    QObject::connect(ui->raBtnZm    , SIGNAL(clicked()), this, SLOT(raBtnZmPressed()));
    QObject::connect(ui->raBtnRollp , SIGNAL(clicked()), this, SLOT(raBtnRollpPressed()));
    QObject::connect(ui->raBtnRollm , SIGNAL(clicked()), this, SLOT(raBtnRollmPressed()));
    QObject::connect(ui->raBtnPitchp, SIGNAL(clicked()), this, SLOT(raBtnPitchpPressed()));
    QObject::connect(ui->raBtnPitchm, SIGNAL(clicked()), this, SLOT(raBtnPitchmPressed()));
    QObject::connect(ui->raBtnYawp  , SIGNAL(clicked()), this, SLOT(raBtnYawpPressed()));
    QObject::connect(ui->raBtnYawm  , SIGNAL(clicked()), this, SLOT(raBtnYawmPressed()));
    QObject::connect(ui->raTxtAngles1, SIGNAL(valueChanged(double)), this, SLOT(raSbAnglesValueChanged(double)));
    QObject::connect(ui->raTxtAngles2, SIGNAL(valueChanged(double)), this, SLOT(raSbAnglesValueChanged(double)));
    QObject::connect(ui->raTxtAngles3, SIGNAL(valueChanged(double)), this, SLOT(raSbAnglesValueChanged(double)));
    QObject::connect(ui->raTxtAngles4, SIGNAL(valueChanged(double)), this, SLOT(raSbAnglesValueChanged(double)));
    QObject::connect(ui->raTxtAngles5, SIGNAL(valueChanged(double)), this, SLOT(raSbAnglesValueChanged(double)));
    QObject::connect(ui->raTxtAngles6, SIGNAL(valueChanged(double)), this, SLOT(raSbAnglesValueChanged(double)));
    QObject::connect(ui->raTxtAngles7, SIGNAL(valueChanged(double)), this, SLOT(raSbAnglesValueChanged(double)));
    QObject::connect(ui->raTxtAnglesG, SIGNAL(valueChanged(double)), this, SLOT(raSbGripperValueChanged(double)));
    QObject::connect(ui->raTxtArticularGoal, SIGNAL(returnPressed()), this, SLOT(raTxtArticularGoalReturnPressed()));
    QObject::connect(ui->raTxtCartesianGoal, SIGNAL(returnPressed()), this, SLOT(raTxtCartesianGoalReturnPressed()));
    
    QObject::connect(ui->hdTxtPan, SIGNAL(valueChanged(double)), this, SLOT(hdSbHeadValueChanged(double)));
    QObject::connect(ui->hdTxtTilt, SIGNAL(valueChanged(double)), this, SLOT(hdSbHeadValueChanged(double)));

    QObject::connect(ui->spgTxtSay, SIGNAL(returnPressed()), this, SLOT(spgTxtSayReturnPressed()));
    QObject::connect(ui->sprTxtFakeRecog, SIGNAL(returnPressed()), this, SLOT(sprTxtFakeRecogReturnPressed()));

    QObject::connect(ui->visBtnFindLines, SIGNAL(clicked()), this, SLOT(visFindLinesClicked()));
    QObject::connect(ui->visBtnFindHoriPlanes, SIGNAL(clicked()), this, SLOT(visFindHoriPlanesClicked()));
    QObject::connect(ui->visTxtTrainObject, SIGNAL(returnPressed()), this, SLOT(visTrainObjectReturnPressed()));
    
    

    QObject::connect(ui->tksTxtTakeObject, SIGNAL(returnPressed()), this, SLOT(visTakeObjectReturnPressed()));
    

    
    QObject::connect(ui->visBtnRecogObjects, SIGNAL(clicked()), this, SLOT(visRecognizeObjectsClicked()));
    QObject::connect(ui->visTxtRecognizeObject, SIGNAL(returnPressed()), this, SLOT(visRecognizeObjectReturnPressed()));
    QObject::connect(ui->visBtnPointsAbovePlane, SIGNAL(clicked()), this, SLOT(visGetPointsAbovePlaneClicked()));
    QObject::connect(ui->visBtnEnableHumanPose, SIGNAL(clicked()), this, SLOT(visEnableHumanPoseClicked()));

    QObject::connect(ui->hriBtnEnableFollowHuman, SIGNAL(clicked()), this, SLOT(hriEnableFollowHumanClicked()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setRosNode(QtRosNode* qtRosNode)
{
    this->qtRosNode = qtRosNode;

    //Connect signals from QtRosNode to MainWindow
    //For example, when ros finishes or when a rostopic is received
    QObject::connect(qtRosNode, SIGNAL(onRosNodeFinished()), this, SLOT(close()));
    QObject::connect(qtRosNode, SIGNAL(updateGraphics()), this, SLOT(updateGraphicsReceived()));
}

void MainWindow::setYamlParser(YamlParser* yamlParser)
{
    this->yamlParser = yamlParser;
}

//
//SLOTS FOR SIGNALS EMITTED IN THE MAINWINDOW
//
void MainWindow::closeEvent(QCloseEvent *event)
{
    this->qtRosNode->gui_closed = true;
    this->qtRosNode->wait();
    //event->accept();
}

void MainWindow::initArmsGuiElements(std_msgs::Float64MultiArray::ConstPtr la_q0, std_msgs::Float64MultiArray::ConstPtr ra_q0)
{
    if(la_q0 == NULL || ra_q0 == NULL)
    {
        std::cout << "JustinaGUI->WARNING!!! No arms initial position received!!! OMG! " << std::endl;
        return;
    }
    ui->laGbArticular->setEnabled(false);
    ui->laTxtAngles1->setValue(la_q0->data[0]);
    ui->laTxtAngles2->setValue(la_q0->data[1]);
    ui->laTxtAngles3->setValue(la_q0->data[2]);
    ui->laTxtAngles4->setValue(la_q0->data[3]);
    ui->laTxtAngles5->setValue(la_q0->data[4]);
    ui->laTxtAngles6->setValue(la_q0->data[5]);
    ui->laTxtAngles7->setValue(la_q0->data[6]);
    ui->laGbArticular->setEnabled(true);
    ui->raGbArticular->setEnabled(false);
    ui->raTxtAngles1->setValue(ra_q0->data[0]);
    ui->raTxtAngles2->setValue(ra_q0->data[1]);
    ui->raTxtAngles3->setValue(ra_q0->data[2]);
    ui->raTxtAngles4->setValue(ra_q0->data[3]);
    ui->raTxtAngles5->setValue(ra_q0->data[4]);
    ui->raTxtAngles6->setValue(ra_q0->data[5]);
    ui->raTxtAngles7->setValue(ra_q0->data[6]);
    ui->raGbArticular->setEnabled(true);
}

//
//SLOTS FOR SIGNALS EMITTED IN THE QTROSNODE
//

void MainWindow::updateGraphicsReceived()
{
    //pmCamera.loadFromData(qtRosNode->imgCompressed.data(), qtRosNode->imgCompressed.size(), "JPG");
    //giCamera->setPixmap(pmCamera);
    ui->laLblCurrentQ1->setText(QString::number(qtRosNode->la_current_q[0], 'f',3));
    ui->laLblCurrentQ2->setText(QString::number(qtRosNode->la_current_q[1], 'f',3));
    ui->laLblCurrentQ3->setText(QString::number(qtRosNode->la_current_q[2], 'f',3));
    ui->laLblCurrentQ4->setText(QString::number(qtRosNode->la_current_q[3], 'f',3));
    ui->laLblCurrentQ5->setText(QString::number(qtRosNode->la_current_q[4], 'f',3));
    ui->laLblCurrentQ6->setText(QString::number(qtRosNode->la_current_q[5], 'f',3));
    ui->laLblCurrentQ7->setText(QString::number(qtRosNode->la_current_q[6], 'f',3));

    ui->laLblCurrentX    ->setText(QString::number(qtRosNode->la_current_cartesian[0], 'f',3));
    ui->laLblCurrentY    ->setText(QString::number(qtRosNode->la_current_cartesian[1], 'f',3));
    ui->laLblCurrentZ    ->setText(QString::number(qtRosNode->la_current_cartesian[2], 'f',3));
    ui->laLblCurrentRoll ->setText(QString::number(qtRosNode->la_current_cartesian[3], 'f',3));
    ui->laLblCurrentPitch->setText(QString::number(qtRosNode->la_current_cartesian[4], 'f',3));
    ui->laLblCurrentYaw  ->setText(QString::number(qtRosNode->la_current_cartesian[5], 'f',3));

    ui->raLblCurrentQ1->setText(QString::number(qtRosNode->ra_current_q[0], 'f',3));
    ui->raLblCurrentQ2->setText(QString::number(qtRosNode->ra_current_q[1], 'f',3));
    ui->raLblCurrentQ3->setText(QString::number(qtRosNode->ra_current_q[2], 'f',3));
    ui->raLblCurrentQ4->setText(QString::number(qtRosNode->ra_current_q[3], 'f',3));
    ui->raLblCurrentQ5->setText(QString::number(qtRosNode->ra_current_q[4], 'f',3));
    ui->raLblCurrentQ6->setText(QString::number(qtRosNode->ra_current_q[5], 'f',3));
    ui->raLblCurrentQ7->setText(QString::number(qtRosNode->ra_current_q[6], 'f',3));

    ui->raLblCurrentX    ->setText(QString::number(qtRosNode->ra_current_cartesian[0], 'f',3));
    ui->raLblCurrentY    ->setText(QString::number(qtRosNode->ra_current_cartesian[1], 'f',3));
    ui->raLblCurrentZ    ->setText(QString::number(qtRosNode->ra_current_cartesian[2], 'f',3));
    ui->raLblCurrentRoll ->setText(QString::number(qtRosNode->ra_current_cartesian[3], 'f',3));
    ui->raLblCurrentPitch->setText(QString::number(qtRosNode->ra_current_cartesian[4], 'f',3));
    ui->raLblCurrentYaw  ->setText(QString::number(qtRosNode->ra_current_cartesian[5], 'f',3));

    ui->laLblVoltage->setText(QString::number(qtRosNode->la_voltage,'f',1));
    ui->raLblVoltage->setText(QString::number(qtRosNode->ra_voltage,'f',1));
    ui->laPbVoltage ->setValue(qtRosNode->la_voltage*10);
    ui->raPbVoltage ->setValue(qtRosNode->ra_voltage*10);
    if(qtRosNode->la_voltage<10.5)
        ui->laLblVoltage->setStyleSheet("QLabel{ color: red; font: 24pt;}");
    else
        ui->laLblVoltage->setStyleSheet("QLabel{ color : black; font: 24pt;}");
    if(qtRosNode->ra_voltage<10.5)
        ui->raLblVoltage->setStyleSheet("QLabel{ color: red; font: 24pt;}");
    else
        ui->raLblVoltage->setStyleSheet("QLabel{ color: black; font: 24pt;}");

    ui->sprLblLastRecog->setText(QString::fromStdString(qtRosNode->spr_recognized));
}

void MainWindow::btnFwdPressed()
{
    qtRosNode->start_publishing_cmd_vel(0.3, 0, 0);
}

void MainWindow::btnFwdReleased()
{
    qtRosNode->stop_publishing_cmd_vel();
}

void MainWindow::btnBwdPressed()
{
    qtRosNode->start_publishing_cmd_vel(-0.3, 0, 0);
}

void MainWindow::btnBwdReleased()
{
    qtRosNode->stop_publishing_cmd_vel();
}

void MainWindow::btnLeftPressed()
{
    qtRosNode->start_publishing_cmd_vel(0, 0.3, 0);
}

void MainWindow::btnLeftReleased()
{
    qtRosNode->stop_publishing_cmd_vel();
}

void MainWindow::btnRightPressed()
{
    qtRosNode->start_publishing_cmd_vel(0, -0.3, 0);
}

void MainWindow::btnRightReleased()
{
    qtRosNode->stop_publishing_cmd_vel();
}

void MainWindow::btnTurnLeftPressed()
{
    qtRosNode->start_publishing_cmd_vel(0, 0, 0.5);
}

void MainWindow::btnTurnLeftReleased()
{
    qtRosNode->stop_publishing_cmd_vel();
}

void MainWindow::btnTurnRightPressed()
{
    qtRosNode->start_publishing_cmd_vel(0, 0, -0.5);
}

void MainWindow::btnTurnRightReleased()
{
    qtRosNode->stop_publishing_cmd_vel();
}

void MainWindow::btnCmdVelPressed()
{
    std::stringstream ssLinearX(ui->txtLinearX->text().toStdString());
    std::stringstream ssLinearY(ui->txtLinearY->text().toStdString());
    std::stringstream ssAngular(ui->txtAngular->text().toStdString());
    float linearX = 0;
    float linearY = 0;
    float angular = 0;
    bool correct_format = true;
    if(!(ssLinearX >> linearX))
    {
        ui->txtLinearX->setText("Invalid format");
        correct_format = false;
    }
    if(!(ssLinearY >> linearY))
    {
        ui->txtLinearY->setText("Invalid format");
        correct_format = false;
    }
    if(!(ssAngular >> angular))
    {
        ui->txtAngular->setText("Invalid format");
        correct_format = false;
    }
    if(correct_format)
        qtRosNode->start_publishing_cmd_vel(linearX, linearY, angular);
}

void MainWindow::btnCmdVelReleased()
{
    qtRosNode->stop_publishing_cmd_vel();
}

void MainWindow::navBtnCalcPath_pressed()
{
    float startX = 0;
    float startY = 0;
    float startA = 0;
    float goalX = 0;
    float goalY = 0;
    float goalA = 0;
    std::vector<std::string> parts;

    std::string str = this->ui->navTxtStartPose->text().toStdString();
    boost::algorithm::to_lower(str);
    boost::split(parts, str, boost::is_any_of(" ,\t\r\n"), boost::token_compress_on);
    if(str.compare("") == 0 || str.compare("robot") == 0) //take robot pose as start position
    {
        this->ui->navTxtStartPose->setText("Robot");
        qtRosNode->get_robot_pose(startX, startY, startA);
    }
    else if(parts.size() >= 2) //Given data correspond to numbers
    {
        std::stringstream ssStartX(parts[0]);
        std::stringstream ssStartY(parts[1]);
        if(!(ssStartX >> startX) || !(ssStartY >> startY))
        {
            this->ui->navTxtStartPose->setText("Invalid format");
            return;
        }
    }
    else
    {
	//this->ui->navTxtStartPose->setText("Invalid format");
	//return;
        if(!qtRosNode->call_known_location(this->ui->navTxtStartPose->text().toStdString(), startX, startY, startA))
        {
            this->ui->navTxtStartPose->setText("Unknown location");
            return;
        }
    }
	
    str = this->ui->navTxtGoalPose->text().toStdString();
    boost::algorithm::to_lower(str);
    boost::split(parts, str, boost::is_any_of(" ,\t\r\n"), boost::token_compress_on);
    if(parts.size() >= 2)
    {
        std::stringstream ssGoalX(parts[0]);
        std::stringstream ssGoalY(parts[1]);
        if(!(ssGoalX >> goalX) || !(ssGoalY >> goalY))
        {
            this->ui->navTxtGoalPose->setText("Invalid format");
            return;
        }
        if(parts.size() >= 3)
	{
	    std::stringstream ssGoalA(parts[2]);
	    if(!(ssGoalA >> goalA))
	    {
		this->ui->navTxtGoalPose->setText("Invalid Format");
		return;
	    }
	}
    }
    else
    {
	//this->ui->navTxtGoalPose->setText("Invalid format");
	//return;
        if(!qtRosNode->call_known_location(this->ui->navTxtGoalPose->text().toStdString(), goalX, goalY, goalA))
        {
            this->ui->navTxtGoalPose->setText("Unknown location");
            return;
        }
    }
    qtRosNode->publish_goal_location(goalX, goalY, goalA);
}

void MainWindow::navBtnExecPath_pressed()
{
    this->navBtnCalcPath_pressed();
}

void MainWindow::torSbPosValueChanged(double d)
{
    qtRosNode->publish_torso_position(ui->torTxtPos->value());
}

/*
 * LEFT ARM CONTROLS
 */
void MainWindow::laBtnXpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[0] += 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}
      
void MainWindow::laBtnXmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[0] -= 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::laBtnYpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[1] += 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::laBtnYmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[1] -= 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::laBtnZpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[2] += 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::laBtnZmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[2] -= 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::laBtnRollpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[3] += 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::laBtnRollmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[3] -= 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::laBtnPitchpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[4] += 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::laBtnPitchmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[4] -= 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::laBtnYawpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[5] += 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::laBtnYawmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->la_current_cartesian;
    goal_cartesian[5] -= 0.05;
    la_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::la_get_IK_and_update_ui(std::vector<double> cartesian)
{
    trajectory_msgs::JointTrajectory q_trajectory;
    std::vector<double> q;
    bool success;
    if(ui->laRbSendTrajectory->isChecked())
    {
        success = qtRosNode->call_la_ik_trajectory(cartesian, q_trajectory);
        if(success)
        {
            q.resize(7);
            for(int i=0; i<7; i++)
                q[i] = q_trajectory.points[q_trajectory.points.size() - 1].positions[i];
        }
    }
    else
        success = qtRosNode->call_la_ik_pose(cartesian, q);
    if(!success)
    {
        std::cout << "JustinaGUI.->Cannot calculate inverse kinematics for left arm." << std::endl;
        return;
    }
    ui->laGbArticular->setEnabled(false);
    ui->laTxtAngles1->setValue(q[0]);
    ui->laTxtAngles2->setValue(q[1]);
    ui->laTxtAngles3->setValue(q[2]);
    ui->laTxtAngles4->setValue(q[3]);
    ui->laTxtAngles5->setValue(q[4]);
    ui->laTxtAngles6->setValue(q[5]);
    ui->laTxtAngles7->setValue(q[6]);
    ui->laGbArticular->setEnabled(true);
    
    if(ui->laRbSendTrajectory->isChecked()) qtRosNode->publish_la_goal_trajectory(q_trajectory);
    else qtRosNode->publish_la_goal_angles(q[0], q[1], q[2], q[3], q[4], q[5], q[6]);
}

void MainWindow::laSbAnglesValueChanged(double d)
{
    if(ui->laGbArticular->isEnabled())
    {
        std::vector<double> goal_q;
        goal_q.resize(7);
        goal_q[0] = ui->laTxtAngles1->value();
        goal_q[1] = ui->laTxtAngles2->value();
        goal_q[2] = ui->laTxtAngles3->value();
        goal_q[3] = ui->laTxtAngles4->value();
        goal_q[4] = ui->laTxtAngles5->value();
        goal_q[5] = ui->laTxtAngles6->value();
        goal_q[6] = ui->laTxtAngles7->value();
        if(ui->laRbSendTrajectory->isChecked())
        {
            trajectory_msgs::JointTrajectory trj;
            if(!qtRosNode->call_get_polynomial_traj(qtRosNode->la_current_q, goal_q, trj))
                return;
            qtRosNode->publish_la_goal_trajectory(trj);
        }
        else 
            qtRosNode->publish_la_goal_angles(ui->laTxtAngles1->value(), ui->laTxtAngles2->value(), ui->laTxtAngles3->value(),
                                              ui->laTxtAngles4->value(), ui->laTxtAngles5->value(), ui->laTxtAngles6->value(),
                                              ui->laTxtAngles7->value());
    }
}

void MainWindow::laSbGripperValueChanged(double d)
{
    if(ui->laGbArticular->isEnabled())
        qtRosNode->publish_la_grip_angles(ui->laTxtAnglesG->value()); 
}

void MainWindow::laTxtArticularGoalReturnPressed()
{
    std::vector<std::string> parts;
    std::vector<double> q = qtRosNode->la_current_q;
    std::string str = this->ui->laTxtArticularGoal->text().toStdString();
    
    YAML::Node yaml_node = yamlParser->nodeLaPredefined[str];
    if(yaml_node)
        for(int i=0; i<7;  i++)
            q[i] = yaml_node[i].as<float>();
    else
    {
        boost::split(parts, str, boost::is_any_of(" ,\t\r\n"), boost::token_compress_on);
        for(size_t i=0; i < parts.size() && i < 7; i++)
        {
            std::stringstream ss(parts[i]);
            if(!(ss >> q[i]))
                q[i] = qtRosNode->la_current_q[i];
        }
    }
    ui->laGbArticular->setEnabled(false);
    ui->laTxtAngles1->setValue(q[0]);
    ui->laTxtAngles2->setValue(q[1]);
    ui->laTxtAngles3->setValue(q[2]);
    ui->laTxtAngles4->setValue(q[3]);
    ui->laTxtAngles5->setValue(q[4]);
    ui->laTxtAngles6->setValue(q[5]);
    ui->laTxtAngles7->setValue(q[6]);
    ui->laGbArticular->setEnabled(true);
    if(ui->laRbSendTrajectory->isChecked())
    {
        trajectory_msgs::JointTrajectory trj;
        if(!qtRosNode->call_get_polynomial_traj(qtRosNode->la_current_q, q, trj))
            return;
        qtRosNode->publish_la_goal_trajectory(trj);
    }
    else 
        qtRosNode->publish_la_goal_angles(q[0], q[1], q[2], q[3], q[4], q[5], q[6]);
    
    
}

void MainWindow::laTxtCartesianGoalReturnPressed()
{
    std::vector<std::string> parts;
    std::string str = this->ui->laTxtCartesianGoal->text().toStdString();
    boost::split(parts, str, boost::is_any_of(" ,\t\r\n"), boost::token_compress_on);
    std::vector<double> cartesian = qtRosNode->la_current_cartesian;
    for(size_t i=0; i < parts.size() && i < 6; i++)
    {
        boost::algorithm::to_lower(parts[i]);
        if(parts[i] == "nan")
            cartesian[i] = std::nan("1");
        else{
            std::stringstream ss(parts[i]);
            if(!(ss >> cartesian[i]))
                cartesian[i] = qtRosNode->la_current_cartesian[i];
        }
    }
    la_get_IK_and_update_ui(cartesian);
}

/*
 * RIGHT ARM CONTROLS
 */
void MainWindow::raBtnXpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[0] += 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}
      
void MainWindow::raBtnXmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[0] -= 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::raBtnYpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[1] += 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::raBtnYmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[1] -= 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::raBtnZpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[2] += 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::raBtnZmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[2] -= 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::raBtnRollpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[3] += 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::raBtnRollmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[3] -= 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::raBtnPitchpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[4] += 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::raBtnPitchmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[4] -= 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::raBtnYawpPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[5] += 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::raBtnYawmPressed()
{
    std::vector<double> goal_cartesian = qtRosNode->ra_current_cartesian;
    goal_cartesian[5] -= 0.05;
    ra_get_IK_and_update_ui(goal_cartesian);
}

void MainWindow::ra_get_IK_and_update_ui(std::vector<double> cartesian)
{
    trajectory_msgs::JointTrajectory q_trajectory;
    std::vector<double> q;
    bool success;
    if(ui->raRbSendTrajectory->isChecked())
    {
        success = qtRosNode->call_ra_ik_trajectory(cartesian, q_trajectory);
        if(success)
        {
            q.resize(7);
            for(int i=0; i<7; i++)
                q[i] = q_trajectory.points[q_trajectory.points.size() - 1].positions[i];
        }
    }
    else
        success = qtRosNode->call_ra_ik_pose(cartesian, q);
    if(!success)
    {
        std::cout << "JustinaGUI.->Cannot calculate inverse kinematics for right arm." << std::endl;
        return;
    }
    ui->raGbArticular->setEnabled(false);
    ui->raTxtAngles1->setValue(q[0]);
    ui->raTxtAngles2->setValue(q[1]);
    ui->raTxtAngles3->setValue(q[2]);
    ui->raTxtAngles4->setValue(q[3]);
    ui->raTxtAngles5->setValue(q[4]);
    ui->raTxtAngles6->setValue(q[5]);
    ui->raTxtAngles7->setValue(q[6]);
    ui->raGbArticular->setEnabled(true);
    
    if(ui->raRbSendTrajectory->isChecked()) qtRosNode->publish_ra_goal_trajectory(q_trajectory);
    else qtRosNode->publish_ra_goal_angles(q[0], q[1], q[2], q[3], q[4], q[5], q[6]);
}

void MainWindow::raSbAnglesValueChanged(double d)
{
    if(ui->raGbArticular->isEnabled())
    {
        std::vector<double> goal_q;
        goal_q.resize(7);
        goal_q[0] = ui->raTxtAngles1->value();
        goal_q[1] = ui->raTxtAngles2->value();
        goal_q[2] = ui->raTxtAngles3->value();
        goal_q[3] = ui->raTxtAngles4->value();
        goal_q[4] = ui->raTxtAngles5->value();
        goal_q[5] = ui->raTxtAngles6->value();
        goal_q[6] = ui->raTxtAngles7->value();
        if(ui->raRbSendTrajectory->isChecked())
        {
            trajectory_msgs::JointTrajectory trj;
            if(!qtRosNode->call_get_polynomial_traj(qtRosNode->ra_current_q, goal_q, trj))
                return;
            qtRosNode->publish_ra_goal_trajectory(trj);
        }
        else 
            qtRosNode->publish_ra_goal_angles(ui->raTxtAngles1->value(), ui->raTxtAngles2->value(), ui->raTxtAngles3->value(),
                                              ui->raTxtAngles4->value(), ui->raTxtAngles5->value(), ui->raTxtAngles6->value(),
                                              ui->raTxtAngles7->value());
    }
}

void MainWindow::raSbGripperValueChanged(double d)
{
    if(ui->raGbArticular->isEnabled())
        qtRosNode->publish_ra_grip_angles(ui->raTxtAnglesG->value()); 
}

void MainWindow::raTxtArticularGoalReturnPressed()
{
    std::vector<std::string> parts;
    std::vector<double> q = qtRosNode->ra_current_q;
    std::string str = this->ui->raTxtArticularGoal->text().toStdString();
    
    YAML::Node yaml_node = yamlParser->nodeRaPredefined[str];
    if(yaml_node)
        for(int i=0; i<7;  i++)
            q[i] = yaml_node[i].as<float>();
    else
    {
        boost::split(parts, str, boost::is_any_of(" ,\t\r\n"), boost::token_compress_on);
        for(size_t i=0; i < parts.size() && i < 7; i++)
        {
            std::stringstream ss(parts[i]);
            if(!(ss >> q[i]))
                q[i] = qtRosNode->ra_current_q[i];
        }
    }
    
    ui->raGbArticular->setEnabled(false);
    ui->raTxtAngles1->setValue(q[0]);
    ui->raTxtAngles2->setValue(q[1]);
    ui->raTxtAngles3->setValue(q[2]);
    ui->raTxtAngles4->setValue(q[3]);
    ui->raTxtAngles5->setValue(q[4]);
    ui->raTxtAngles6->setValue(q[5]);
    ui->raTxtAngles7->setValue(q[6]);
    ui->raGbArticular->setEnabled(true);
    if(ui->raRbSendTrajectory->isChecked())
    {
        trajectory_msgs::JointTrajectory trj;
        if(!qtRosNode->call_get_polynomial_traj(qtRosNode->ra_current_q, q, trj))
            return;
        qtRosNode->publish_ra_goal_trajectory(trj);
    }
    else 
        qtRosNode->publish_ra_goal_angles(q[0], q[1], q[2], q[3], q[4], q[5], q[6]);
}

void MainWindow::raTxtCartesianGoalReturnPressed()
{
    std::vector<std::string> parts;
    std::string str = this->ui->raTxtCartesianGoal->text().toStdString();
    boost::split(parts, str, boost::is_any_of(" ,\t\r\n"), boost::token_compress_on);
    std::vector<double> cartesian = qtRosNode->ra_current_cartesian;
    for(size_t i=0; i < parts.size() && i < 6; i++)
    {
        std::stringstream ss(parts[i]);
        if(!(ss >> cartesian[i]))
            cartesian[i] = qtRosNode->ra_current_cartesian[i];
    }
    ra_get_IK_and_update_ui(cartesian);
}

/*
 * HEAD CONTROLS
 */
void MainWindow::hdSbHeadValueChanged(double d)
{
    qtRosNode->publish_head_angles(ui->hdTxtPan->value(), ui->hdTxtTilt->value());
}

void MainWindow::spgTxtSayReturnPressed()
{
    qtRosNode->say(ui->spgTxtSay->text().toStdString());
}

void MainWindow::sprTxtFakeRecogReturnPressed()
{
    qtRosNode->publish_fake_speech_recog(ui->sprTxtFakeRecog->text().toStdString());
}

void MainWindow::visFindLinesClicked()
{
    qtRosNode->call_find_lines();
}

void MainWindow::visFindHoriPlanesClicked()
{
    qtRosNode->call_find_horizontal_planes();
}

void MainWindow::visTrainObjectReturnPressed()
{
    qtRosNode->call_train_object(ui->visTxtTrainObject->text().toStdString());
}

void MainWindow::visRecognizeObjectReturnPressed()
{
    qtRosNode->call_recognize_object(ui->visTxtRecognizeObject->text().toStdString());
}

void MainWindow::visRecognizeObjectsClicked()
{
    qtRosNode->call_recognize_objects();
}

void MainWindow::visTakeObjectReturnPressed()
{
    qtRosNode->call_take_object(ui->tksTxtTakeObject->text().toStdString());
}


void MainWindow::visGetPointsAbovePlaneClicked()
{
    qtRosNode->call_get_points_above_plane();
}

void MainWindow::visEnableHumanPoseClicked()
{
    bool enable = ui->visBtnEnableHumanPose->text().toStdString().find("Enable") != std::string::npos;
    if(enable) ui->visBtnEnableHumanPose->setText("Disable\nHuman Pose Det");
    else ui->visBtnEnableHumanPose->setText("Enable\nHuman Pose Det");
    qtRosNode->publish_enable_human_pose_detection(enable);
}

void MainWindow::hriEnableFollowHumanClicked()
{
    bool enable = ui->hriBtnEnableFollowHuman->text().toStdString().find("Enable") != std::string::npos;
    if(enable) ui->hriBtnEnableFollowHuman->setText("Disable\nFollow Human");
    else ui->hriBtnEnableFollowHuman->setText("Enable\nFollow Human");
    qtRosNode->publish_enable_human_following(enable);
}
