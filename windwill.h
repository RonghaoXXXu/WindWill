#ifndef WINDWILL_H_
#define WINDWILL_H_

#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#define BLUE 1
#define RED 0

struct WindParams{

    int max_imgthresold;
    int min_imgthresold;
    int infer_frame_number;
    int infer_frame_number1;
    int infer_frame_number2;

    float Armor_maxHWRation;
//    float Armor_maxArea;
//    float Armor_minArea;
    float Leaf_minArea;

    bool Enemy_color;

    int Min_n;

    WindParams()
    {

        max_imgthresold=100;
        min_imgthresold=60;
        infer_frame_number=100;
        infer_frame_number1=infer_frame_number/3;
        infer_frame_number2=infer_frame_number*2/3;

        Armor_maxHWRation=0.7153846;
//        Armor_maxArea=800;
//        Armor_minArea=500;
        Leaf_minArea=3000;//重要参数

        Enemy_color=RED;

        Min_n=600;

    }

};

class Windwill{

public:

    void Action();

    void Pre_process();//Leaf Armor PreImg

    void Get_target_Armor();

    Mat getSvmInput(Mat &input);
    vector<float> stander(Mat &im);

    void Circle_bound();
    void Direction();

    void precise();

    Point2f Min_Motion_Predict(Point2f point1);
    Point2f Max_Motion_Predict(Point2f point1);

    float getDistence(Point2f &a,Point2f &b){
        float dis;
        dis=(float)sqrt(pow(double(a.x-b.x),2)+pow(double(a.y-b.y),2));
        return dis;
    }

    float funct(float t){
        return 0.785*sin(1.884*t)+1.035;
    }

    void Windclear(){
        process_time=0;
        vector<Point2f>().swap(Armor_Centers);
        Radius=0;
        R_center=Point2f(0,0);

        Armor_temp_center=Point2f(0,0);
        Armor_Aim_center=Point2f(0,0);

        vector<Point2f>().swap(points_2d_temp);
        vector<Point2f>().swap(points_2d_Aim);

        direction=CLKNONE;
        Wind_flag=WIND_NONE;
    }

    Mat src;

/**********电控发视觉*********/
    float shoot_time;//射击时间
    float process_time;//Max的t时间

    int Wind_flag=WIND_NONE;//能量机关模式
    int Shoot_flag=RESTORE_AIM;//射击情况

/*********视觉发电控*********/

    float yaw;
    float pitch;
    float distance;




    typedef enum {
        WIND_NONE =0,
        WIND_MAX,
        WIND_MIN
    }WIND_FlAG;

    typedef enum {
        CLKNONE=0,
        CLKWISE,
        CCLKWISE
    }Windirection;

    typedef enum {
        RESET_NEW =0,
        FOLLOW_AGAIN,
        RESTORE_AIM
    }SHOOT_FlAG;

    int direction=CLKNONE;


private:

    WindParams Params;

    Mat temp,temp1,temp2;
    Mat Leaf_preImg;
    Mat Armor_preImg;
    Mat draw_show=Mat::zeros(Size(800,800),CV_8UC3);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    Ptr<ml::SVM> svm=ml::SVM::load("svm4_9.xml");

    vector<Point2f> Armor_Centers;
    vector<RotatedRect> Armor_rect;

    double Radius;
    Point2f R_center;

    Point2f Armor_temp_center;
    Point2f Armor_Aim_center;

    vector<Point2f> points_2d_temp;
    vector<Point2f> points_2d_Aim;


};

#endif
