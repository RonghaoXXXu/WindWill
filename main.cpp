#include <iostream>
#include <opencv2/opencv.hpp>
#include <windmill.h>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap("2.mov");
    WindMill wind;
    WindMill::armorData data;
    Point2f pt;
    int status;
    while(cap.isOpened()){
        cap>>wind.src;
        resize(wind.src,wind.src,Size(800,400));
//        wind.getArmorCenter(wind.src,data);
        wind.detect(wind.src,1,pt,status);
        if(pt!=Point2f(0,0))
            cout<<pt<<endl;
        waitKey(50);
    }
    return 0;
}
