#include "anglesolver.h"
#include "windwill.h"

/*void AngleSolver::onePointSolution(const vector<Point2f> centerPoint)
{
    double fx = _params.CameraIntrinsicMatrix.at<double>(0,0);
    double fy = _params.CameraIntrinsicMatrix.at<double>(1,1);
    double cx = _params.CameraIntrinsicMatrix.at<double>(0,2);
    double cy = _params.CameraIntrinsicMatrix.at<double>(1,2);

    vector<Point2f> dstPoint;
    //单点矫正
    undistortPoints(centerPoint,dstPoint,_params.CameraIntrinsicMatrix,
                    _params.DistortionCoefficient,noArray(),_params.CameraIntrinsicMatrix);
    Point2f pnt = dstPoint.front();//返回dstPoint中的第一个元素
    //去畸变后的比值，根据像素坐标系与世界坐标系的关系得出,pnt的坐标就是在整幅图像中的坐标
    double rxNew=(pnt.x-cx)/fx;
    double ryNew=(pnt.y-cy)/fy;

    yawErr = atan(rxNew)/CV_PI*180;//转换为角度
    pitchErr = atan(ryNew)/CV_PI*180;//转换为角度
}*/

void AngleSolver::p4pSolution(const std::vector<cv::Point2f> objectPoints){

//    if(Windwill.Wind_flag == Windwill.WIND_MAX)
//        solvePnP(_params.POINT_3D_OF_ARMOR_BIG,objectPoints,_params.CameraIntrinsicMatrix,
//                 _params.DistortionCoefficient,_rVec,_tVec,false, SOLVEPNP_ITERATIVE);
//    else if(Windwill.Wind_flag == Windwill.WIND_MIN)
//        solvePnP(_params.POINT_3D_OF_ARMOR_SMALL,objectPoints,_params.CameraIntrinsicMatrix,
//                 _params.DistortionCoefficient,_rVec,_tVec,false, SOLVEPNP_ITERATIVE);

    /*****_tVec 相机到云台******/


    _tVec.at<float>(1, 0) -= _params.Y_DISTANCE_BETWEEN_GUN_AND_CAM;
    _tVec.at<float>(2, 0) -= _params.Z_DISTANCE_BETWEEN_MUZZLE_AND_CAM;

    _xErr = atan(_tVec.at<float>(0, 0)/_tVec.at<float>(2, 0))/CV_PI*360;//转换为角度
    _yErr = atan(_tVec.at<float>(1, 0)/_tVec.at<float>(2, 0))/CV_PI*360;//转换为角度

    _euclideanDistance = sqrt(_tVec.at<float>(0, 0)*_tVec.at<float>(0, 0) + _tVec.at<float>(1, 0)*
                              _tVec.at<float>(1, 0) + _tVec.at<float>(2, 0)* _tVec.at<float>(2, 0));
}
//摄像头位置偏移矫正
void AngleSolver::compensateOffset()
{
    /* z of the camera COMS */
    const auto offset_z = 120.0;
    const auto& d = _euclideanDistance;
    const auto theta_y = _xErr / 180 * CV_PI;
    const auto theta_p = _yErr / 180 * CV_PI;
    const auto theta_y_prime = atan((d*sin(theta_y)) / (d*cos(theta_y) + offset_z));
    const auto theta_p_prime = atan((d*sin(theta_p)) / (d*cos(theta_p) + offset_z));
    const auto d_prime = sqrt(pow(offset_z + d * cos(theta_y), 2) + pow(d*sin(theta_y), 2));
    _xErr = theta_y_prime / CV_PI * 180;
    _yErr = theta_p_prime / CV_PI * 180;
    _euclideanDistance = d_prime;
}
//重力补偿
void AngleSolver::compensateGravity()
{
    const auto& theta_p_prime = _yErr / 180 * CV_PI;
    const auto& d_prime = _euclideanDistance;
    const auto& v = _bullet_speed;
    const auto theta_p_prime2 = atan((sin(theta_p_prime) - 0.5*9.8*d_prime / pow(v, 2)) / cos(theta_p_prime));
    _yErr = theta_p_prime2 / CV_PI * 180;
}
