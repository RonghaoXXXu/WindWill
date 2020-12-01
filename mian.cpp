#include "opencv2/opencv.hpp"
#include "windwill.h"
#include <iostream>

using namespace std;
using namespace cv;

int main(){
    Windwill wind;
    VideoCapture cap("6.mov");
    //cap.set(5,15);
    while(cap.isOpened()){
        cap>>wind.src;
        wind.Pre_process();
        wind.Get_target_Armor();
        wind.Circle_bound();
        wind.Direction();
        //if(wind.Find_armor==false) break;
        //wind.Windclear();

        //imshow("sss",wind.src);
        waitKey(1);
    }
    cap.release();
//    wind.src=imread("1.jpg");
//    wind.Pre_process();

    return 0;

}

