#ifndef ANGLESOLVER_H
#define ANGLESOLVER_H

#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

struct AngleSolverParam
{
    cv::Mat CameraIntrinsicMatrix; //相机内参矩阵
    cv::Mat DistortionCoefficient; //相机畸变系数
    //单位为mm
    std::vector<cv::Point3f> POINT_3D_OF_ARMOR_BIG;
    std::vector<cv::Point3f> POINT_3D_OF_ARMOR_SMALL;
    float Y_DISTANCE_BETWEEN_GUN_AND_CAM;//如果摄像头在枪管的上面，这个变量为正
    float Z_DISTANCE_BETWEEN_MUZZLE_AND_CAM;//如果摄像头在枪口的后面，这个变量为正

    AngleSolverParam()
    {
        //大装甲板真实的世界坐标，把大装甲板看做一个平板，Z坐标为0，把大装甲板的中心对准定义的图像中心
        POINT_3D_OF_ARMOR_BIG = {
            cv::Point3f(-117.5f, -63.5f, 0.0f), //tl
            cv::Point3f(117.5f, -63.5f, 0.0f),	//tr
            cv::Point3f(117.5f, 63.5f, 0.0f),	//br
            cv::Point3f(-117.5f, 63.5f, 0.0f)	//bl
        };
        //如上
        POINT_3D_OF_ARMOR_SMALL = {
            cv::Point3f(-70.0f, -62.5f, 0.0f),	//tl
            cv::Point3f(70.0f, -62.5f, 0.0f),	//tr
            cv::Point3f(70.0f, 62.5f, 0.0f),    //br
            cv::Point3f(-70.0f, 62.5f, 0.0f)    //bl
        };

        FileStorage fsRead("camera.xml",FileStorage::READ);

        if(!fsRead.isOpened())
        {
            cout << "failed to open xml" << endl;
            return;
        }

//        fsRead["Y_DISTANCE_BETWEEN_GUN_AND_CAM"] >> Y_DISTANCE_BETWEEN_GUN_AND_CAM;
//        fsRead["Z_DISTANCE_BETWEEN_MUZZLE_AND_CAM"] >> Z_DISTANCE_BETWEEN_MUZZLE_AND_CAM;
        fsRead["Camera_Matrix"] >> CameraIntrinsicMatrix;
        fsRead["Distortion_Coefficients"] >> DistortionCoefficient;
    }
};

class AngleSolver
{
public:
    //利用小孔成像原理进行单点解算，只能得到相对于摄像头中心的转角,没有深度信息，仅需一个点
//    void onePointSolution(const std::vector<cv::Point2f> centerPoint);

    void p4pSolution(const std::vector<cv::Point2f> objectPoints);

    void compensateOffset();
    void compensateGravity();

private:
    AngleSolverParam _params;

    cv::Mat _rVec = cv::Mat::zeros(3, 1, CV_32FC1);//像素坐标系到相机坐标系的旋转矩阵
    cv::Mat _tVec = cv::Mat::zeros(3, 1, CV_32FC1);//像素坐标系到相机坐标系的平移向量

    cv::Mat trans_camera2ptz; //相机坐标系到云台坐标系的平移向量
    cv::Mat rot_camera2ptz; //相机坐标系到云台坐标系的旋转矩阵

    double _euclideanDistance;//x坐标下的差值，y坐标下的差值，欧氏距离
    double _xErr,_yErr; //yaw轴的误差，pitch轴的误差
    double _bullet_speed=25000;
};


#endif // ANGLESOLVER_H
