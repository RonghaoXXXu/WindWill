#ifndef WINDMILL_H
#define WINDMILL_H


#include <bits/stdc++.h>
#include <pthread.h>
#include <assert.h>
#include "opencv2/opencv.hpp"
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace std;

#define NO_TARGET -1
#define MAX_NUM 921600
#define RED 0
#define BLUE 1

class WindMill{

    enum predictMode{
        FIT_CIRCLE = 0,
        PUSH_CIRCLE,
        TANGENT
    };

    enum detectMode{
        RED_ANCLOCK = 0,
        BLUE_ANCLOCK,
        RED_CLOCK ,
        BLUE_CLOCK ,
        RED_STATIC ,
        BLUE_STATIC
    };

    enum shootMode{
        SHOOT_NONE = 0,
        SHOOT_MAX ,
        SHOOT_MIN
    };

public:
    struct armorData{
        Point2f armorCenter;
        Point2f RCenter;
        double Radius;
        double Rect_Radius[4] = {0};
        float angle;
        int quadrant;
        bool isFind;
        double time;
        RotatedRect armorRect;
        armorData(){
            armorCenter = Point2f(0,0);
            RCenter = Point2f(0,0);
            angle = 0;
            quadrant = 0;
            isFind = false;
            time=0;
        }
    };

private:

    struct DectParam{
        int pMode;
        float preAngle;

        int cutLimitedTime = 40;

        int max_imgthresold;
        int min_imgthresold;

        int infer_frame_number;
        int infer_again_number;
        int time_number;
        double shoot_time;

        float Armor_maxHWRation;
        float Leaf_minArea;

        bool Enemy_color;
        int Min_n;//小能量机关的转速

        DectParam(){

            cutLimitedTime = 40;// 400ms

            max_imgthresold = 85;
            min_imgthresold = 70;

            infer_again_number=10;

            Armor_maxHWRation=0.7153846;
            Leaf_minArea=300;

            Enemy_color = BLUE;

            Min_n=600;
            shoot_time=3;//射击时间
        }
    };
public:
    Mat src;

    Mat mat_track;
    Rect2d aim_rect;

    armorData lastData;
    armorData lostData;
    armorData testData;

    Point2f pre_points[4]={Point2f(0,0)};
    Point2f preCenter = Point2f(0,0);

    clock_t start;
    ofstream file;
private:
    // param
    DectParam param;

    // mode init
    int detect_mode = (param.Enemy_color==RED)? RED_STATIC:BLUE_STATIC;
    int shoot_mode = SHOOT_NONE;

    uint frame_cnt = 0;

    // machine learning
    Ptr<ml::SVM> svm = ml::SVM::load("svm.xml");

public:
    //Tracker MOSSE
    Ptr<cv::Tracker> tracker = TrackerCSRT::create();

private:
    float distance(const Point2f pt1,const Point2f pt2){
        return sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x)+(pt1.y - pt2.y)*(pt1.y - pt2.y));
    }
    bool makeRectSafe(const Rect rect,const Size size);
    bool circleLeastFit(const vector<Point2f> &points,Point2f &R_center);
    bool change_angle(const int quadrant,const float angle,float &tran_angle);
public:
    WindMill();
    void clear();
    bool getArmorCenter(const Mat src,armorData &data);
    Mat getSvmInput(Mat &input);
    vector<float> stander(Mat &im);
    bool getDirection();
    void isCut(const armorData new_data,int &status);
    bool predict(const armorData data,Point2f& preCenter,int pMode);
    double Max_W_Function(double t,double fai);
    double Max_Motion_Predict(const armorData new_data);
    void detect(const Mat frame,int Mode,Point2f &pt, RotatedRect &rect, Point2f pts[],int &status);

    void Tracking(const Mat src,Rect2d rect);
    pair<bool,RotatedRect> KalmanFilter_ROI(const armorData new_data);
};
#endif // WINDMILL_H
