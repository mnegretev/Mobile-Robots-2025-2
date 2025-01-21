#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <iostream>
#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsRectItem>
#include <QGraphicsPixmapItem>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/filesystem/path.hpp>
#include "QtRosNode.h"
#include "YamlParser.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    QtRosNode* qtRosNode;
    YamlParser* yamlParser;

    void setRosNode(QtRosNode* qtRosNode);
    void setYamlParser(YamlParser* yamlParser);
    void closeEvent(QCloseEvent *event);
    void initArmsGuiElements(std_msgs::Float64MultiArray::ConstPtr la_q0, std_msgs::Float64MultiArray::ConstPtr ra_q0);

public slots:
    //Slots for signals emitted in the QtRosNode (e.g. a topic is received)
    void updateGraphicsReceived();
    void btnFwdPressed();
    void btnFwdReleased();
    void btnBwdPressed();    
    void btnBwdReleased();
    void btnLeftPressed();
    void btnLeftReleased();
    void btnRightPressed();
    void btnRightReleased();
    void btnTurnLeftPressed();
    void btnTurnLeftReleased();
    void btnTurnRightPressed();
    void btnTurnRightReleased();
    void btnCmdVelPressed();
    void btnCmdVelReleased();

    void navBtnCalcPath_pressed();
    void navBtnExecPath_pressed();

    void torSbPosValueChanged(double d);
    
    void laSbAnglesValueChanged(double d);
    void laSbGripperValueChanged(double d);
    void laTxtArticularGoalReturnPressed();
    void laTxtCartesianGoalReturnPressed();
    void laBtnXpPressed();
    void laBtnXmPressed();
    void laBtnYpPressed();
    void laBtnYmPressed();
    void laBtnZpPressed();
    void laBtnZmPressed();
    void laBtnRollpPressed();
    void laBtnRollmPressed();
    void laBtnPitchpPressed();
    void laBtnPitchmPressed();
    void laBtnYawpPressed();
    void laBtnYawmPressed();
    void la_get_IK_and_update_ui(std::vector<double> cartesian);
    
    void raSbAnglesValueChanged(double d);
    void raSbGripperValueChanged(double d);
    void raTxtArticularGoalReturnPressed();
    void raTxtCartesianGoalReturnPressed();
    void raBtnXpPressed();
    void raBtnXmPressed();
    void raBtnYpPressed();
    void raBtnYmPressed();
    void raBtnZpPressed();
    void raBtnZmPressed();
    void raBtnRollpPressed();
    void raBtnRollmPressed();
    void raBtnPitchpPressed();
    void raBtnPitchmPressed();
    void raBtnYawpPressed();
    void raBtnYawmPressed();
    void ra_get_IK_and_update_ui(std::vector<double> cartesian);

    void hdSbHeadValueChanged(double d);

    void spgTxtSayReturnPressed();
    void sprTxtFakeRecogReturnPressed();

    void visFindLinesClicked();
    void visFindHoriPlanesClicked();
    void visTrainObjectReturnPressed();


    void visTakeObjectReturnPressed();


    void visRecognizeObjectReturnPressed();
    void visRecognizeObjectsClicked();
    void visGetPointsAbovePlaneClicked();
    void visEnableHumanPoseClicked();

    void hriEnableFollowHumanClicked();
private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
