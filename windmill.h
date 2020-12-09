#ifndef WINDMILL_H
#define WINDMILL_H


#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#define NO_TARGET -1
#define MAX_NUM 921600
#define RED 0
#define BLUE 1
//#define GET_ROI

class WindMill{
//    enum binaryMode{
//        BGR = 1,
//        HSV = 2,
//        BGR_useG = 3,
//        OTSU = 4,
//        GRAY = 5,
//        YCrCb = 6,
//        LUV = 7,
//    };

    enum predictMode{
        FIT_CIRCLE = 1,
        PUSH_CIRCLE = 2,
        TANGENT = 3
    };

    enum detectMode{
        RED_ANCLOCK = 3,
        BLUE_ANCLOCK = 4,
        RED_CLOCK = 5,
        BLUE_CLOCK = 6,
        RED_STATIC = 7,
        BLUE_STATIC = 8
    };

    enum shootMode{
        SHOOT_NONE =0,
        SHOOT_MAX =1,
        SHOOT_MIN
    };

public:
    struct armorData{
        Point2f armorCenter;
        Point2f R_center;
        float angle;
        int quadrant;
        bool isFind;
        double time;
        armorData(){
            armorCenter = Point2f(0,0);
            R_center = Point2f(0,0);
            angle = 0;
            quadrant = 0;
            isFind = false;// 0: 未识别，1: 全部识别
            time=0;
        }
    };

private:
    struct DectParam{
        //int bMode;
        int pMode;
        int radius;
        float preAngle;

        int cutLimitedTime = 40;
/****************XRH*****************/
        int max_imgthresold;
        int min_imgthresold;

        int infer_frame_number;
    //    int infer_frame_number1;
    //    int infer_frame_number2;
        int infer_again_number;
        int time_number;
        double shoot_time;

        float Armor_maxHWRation;
        //    float Armor_maxArea;
        //    float Armor_minArea;
        float Leaf_minArea;

        bool Enemy_color;
//        bool FOLLOW_AIM;
        int Min_n;//小能量机关的转速

 /**********************************/
        DectParam(){
            // radius
            radius = 148;
            // mode
            //bMode = GRAY;
            pMode = FIT_CIRCLE;
            cutLimitedTime = 40;// 400ms
            // predict


/****************XRH*****************/
            max_imgthresold=100;
            min_imgthresold=60;

// 大符速度测量
            infer_frame_number=15;//重要参数
    //        infer_frame_number1=infer_frame_number/3;
    //        infer_frame_number2=infer_frame_number*2/3;
            infer_again_number=10;
            time_number=10;//数据传输时间内的帧数

            Armor_maxHWRation=0.7153846;
            //        Armor_maxArea=800;
            //        Armor_minArea=500;
            Leaf_minArea=300;//重要参数

            Enemy_color = RED;
//            FOLLOW_AIM = false;

            Min_n=600;
            shoot_time=0.033;
/***********************************/
        }
    };
public:
    Mat src;
    armorData lastData;
    armorData lostData;
private:
    // param
    DectParam param;

    // init
    int detect_mode = (param.Enemy_color==RED)? RED_STATIC:BLUE_STATIC;
    int shoot_mode = SHOOT_NONE;
    //Mat debug_src;
    uint frame_cnt = 0;
    //bool dirFlag;

    // machine learning
    Ptr<ml::SVM> svm;

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
    //bool forward(const Mat src,Mat &dect_src,Point2f &offset);
    //bool setImage(const Mat src,Mat &dect_src,Point2f &offset);
    //bool setBinary(const Mat src,Mat &binary,int bMode);
    bool getArmorCenter(const Mat src,armorData &data);
    Mat getSvmInput(Mat &input);
    vector<float> stander(Mat &im);
    bool getDirection();
    void isCut(const armorData new_data,int &status);
    bool predict(const armorData data,Point2f& preCenter,int pMode);
    double Max_W_Function(double t,double fai);
    //void Min_Motion_Predict();
    double Max_Motion_Predict(const armorData new_data);
    void detect(const Mat frame,int Mode,Point2f& pt,int& status);
};

#endif // WINDMILL_H
