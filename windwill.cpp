#include "windwill.h"

void Windwill::Action(){
    Pre_process();
    Get_target_Armor();

    precise();
    Circle_bound();
    Direction();

    for(int i=0;i<4;i++)
    {
        if(Wind_flag==WIND_MIN){
            points_2d_Aim.push_back(Min_Motion_Predict(points_2d_temp[i]));
            Armor_Aim_center=Min_Motion_Predict(Armor_temp_center);
        }
        else if(Wind_flag==WIND_MAX){
            points_2d_Aim.push_back(Max_Motion_Predict(points_2d_temp[i]));
            Armor_Aim_center=Max_Motion_Predict(Armor_Aim_center);
        }
    }

}

void Windwill::Pre_process()
{
    vector<Mat> channels;
    Mat color;
    split(src, channels);

    if (Params.Enemy_color == RED)
    {
        color = channels[2] - channels[0];
    }
    else
    {
        color = channels[0] - channels[2];
    }

    threshold(color, temp1, Params.max_imgthresold, 255, THRESH_BINARY);
    threshold(color, temp2, Params.min_imgthresold, 255, THRESH_BINARY);

    add(temp1,temp2,temp);

    int elementSize = 3;
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * elementSize + 1, 2 * elementSize + 1),Size(elementSize,elementSize));
    dilate(temp, temp, element);

    floodFill(temp,Point(0,0),Scalar(0));

    elementSize = 2;
    element = getStructuringElement(MORPH_RECT, Size(2 * elementSize + 1, 2 * elementSize + 1),Size(elementSize,elementSize));
    morphologyEx(temp, temp, MORPH_CLOSE, element);

    Leaf_preImg = temp.clone();
//    Armor_preImg = temp.clone();
//
 //   imshow("temp1",temp1);
 //   imshow("temp2",temp2);
//    imshow("temp",temp);

//    waitKey(0);
}

void Windwill::Get_target_Armor()
{
    findContours(Leaf_preImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    if (hierarchy.size())
    {
        for (int i = 0; (i>=0) && (hierarchy[i][3]< 0); i=hierarchy[i][0])
        {
            RotatedRect leaf_temp_rect = minAreaRect(contours[i]);
            Rect temp1=boundingRect(contours[i]);


            Point2f p[4];
            leaf_temp_rect.points(p);

            Point2f srcRect[4];
            Point2f dstRect[4];

            double width = getDistence(p[0], p[1]);
            double height = getDistence(p[1], p[2]);

            if (width > height)
            {
                srcRect[0] = p[0];
                srcRect[1] = p[1];
                srcRect[2] = p[2];
                srcRect[3] = p[3];
            }
            else
            {
                swap(width, height);
                srcRect[0] = p[1];
                srcRect[1] = p[2];
                srcRect[2] = p[3];
                srcRect[3] = p[0];
            }
            if (contourArea(contours[i])> Params.Leaf_minArea)
            {
                //Find_armor=false;

                dstRect[0] = Point2f(0, 0);
                dstRect[1] = Point2f(width, 0);
                dstRect[2] = Point2f(width, height);
                dstRect[3] = Point2f(0,height);

                Mat transformMat = getPerspectiveTransform(srcRect, dstRect);
                Mat perspectMat;
                warpPerspective(Leaf_preImg, perspectMat, transformMat, Leaf_preImg.size());

                Mat mat;
                mat = perspectMat(Rect(0, 0, width, height));

                Mat svm_mat=getSvmInput(mat);

                if(svm->predict(svm_mat) ==1 && (hierarchy[i][2] > 0))
                {
                    if (hierarchy[hierarchy[i][2]][1] < 0 && hierarchy[hierarchy[i][2]][0] < 0)
                    {
//                        //Find_armor=true;
//                        Mat armor1=Mat(src,temp1);
//                        imshow("father",armor1);

                        for (int i = 0; i < 4; i++)
                            line(src, p[i], p[(i+1)%4], Scalar(0,255,0),2);

                        RotatedRect armor_rect = minAreaRect(contours[hierarchy[i][2]]);
                        Rect temp2=boundingRect(contours[hierarchy[i][2]]);

                        Point2f pnt[4];
                        armor_rect.points(pnt);

                        float width = armor_rect.size.width;
                        float height = armor_rect.size.height;

                        //double area = armor_rect.size.area();

                        if (height > width)
                            swap(width, height);

                        if ((height / width) > Params.Armor_maxHWRation) // Params.Armor_maxArea && area > Params.Armor_minArea)
                            continue;

                        for (int i = 0; i < 4; i++)
                            line(src, pnt[i], pnt[(i+1)%4], Scalar(0,255,0),3);

                        circle(src,armor_rect.center,1,Scalar(0,0,255),8,LINE_8,0);

                        Mat armor2=Mat(src,temp2);
                        imshow("child",src);

                        Armor_rect.push_back(armor_rect);
                        Armor_Centers.push_back(armor_rect.center);
                    }
                }
            }
        }
    }
}

Mat Windwill::getSvmInput(Mat &input)
{
    vector<float> vec=stander(input);
    if(vec.size()!=900) cout<<"wrong1 not 900"<<endl;
    Mat output(1,900,CV_32FC1);

    Mat_<float> p=output;
    int jj=0;
    for(vector<float>::iterator iter=vec.begin();iter!=vec.end();iter++,jj++)
    {
        p(0,jj)=*(iter);
    }
    return output;
}

vector<float> Windwill::stander(Mat &im)
{
    if(im.empty()==1)
        cout<<"filed open"<<endl;
    resize(im,im,Size(48,48));
    vector<float> result;
    HOGDescriptor hog(Size(48,48),Size(16,16),Size(8,8),Size(8,8),9,1,-1,
                      HOGDescriptor::L2Hys,0.2,false,HOGDescriptor::DEFAULT_NLEVELS);
    hog.compute(im,result);
    return result;
}

void Windwill::Circle_bound()
{
/*   if ((int)Armor_Centers.size() == Params.infer_frame_number)
    {
        int n=Armor_Centers.size();
        const int lr = 2;
        vector<float> losses(n);
        vector<float> min_loss(n);
        vector<float> root_val(n);

        float x, y, r;

        for (int i = 0; i < n; i++)
        {
            float loop_loss = 0;
            for (int j = 0; j < n; j++)
            {
                root_val[j] = sqrt(pow((Armor_Centers[j].x - x), 2) + pow((Armor_Centers[j].y - y), 2));
                const float loss = root_val[j] - r;
                losses[j] += loss;
                loop_loss += fabs(loss);
            }
            min_loss[i] = loop_loss;

            if (i > 0 && min_loss[i] < min_loss[i - 1])
                break;

            float gx, gy, gr;

            for (int j = 0; j < n; j++)
            {
                float gxi = (x - Armor_Centers[j].x) / root_val[j];
                if (losses[j] < 0)
                    gxi *= (-1);

                float gyi = (y - Armor_Centers[j].y) / root_val[j];
                if (losses[j] < 0)
                    gyi *= (-1);

                float gri = -1;
                if (losses[j] < 0)
                    gri = 1;

                gx += gxi;
                gy += gyi;
                gr += gri;
            }

            gx /= n;
            gy /= n;
            gr /= n;

            x -= (lr * gx);
            y -= (lr * gy);
            r -= (lr * gr);
        }

        R_center.x = x;
        R_center.y = y;
        Radius = r;
    }
*/

    int iNum = (int)Armor_Centers.size();
    if (iNum == Params.infer_frame_number)
    {
        double X1 = 0.0;
        double Y1 = 0.0;
        double X2 = 0.0;
        double Y2 = 0.0;
        double X3 = 0.0;
        double Y3 = 0.0;
        double X1Y1 = 0.0;
        double X1Y2 = 0.0;
        double X2Y1 = 0.0;
        vector<cv::Point2f>::iterator iter;
        vector<cv::Point2f>::iterator end = Armor_Centers.end();
        for (iter = Armor_Centers.begin(); iter != end; ++iter)
        {
            X1 = X1 + (*iter).x;
            Y1 = Y1 + (*iter).y;
            X2 = X2 + (*iter).x * (*iter).x;
            Y2 = Y2 + (*iter).y * (*iter).y;
            X3 = X3 + (*iter).x * (*iter).x * (*iter).x;
            Y3 = Y3 + (*iter).y * (*iter).y * (*iter).y;
            X1Y1 = X1Y1 + (*iter).x * (*iter).y;
            X1Y2 = X1Y2 + (*iter).x * (*iter).y * (*iter).y;
            X2Y1 = X2Y1 + (*iter).x * (*iter).x * (*iter).y;
        }
        double C = 0.0;
        double D = 0.0;
        double E = 0.0;
        double G = 0.0;
        double H = 0.0;
        double a = 0.0;
        double b = 0.0;
        double c = 0.0;
        C = iNum * X2 - X1 * X1;
        D = iNum * X1Y1 - X1 * Y1;
        E = iNum * X3 + iNum * X1Y2 - (X2 + Y2) * X1;
        G = iNum * Y2 - Y1 * Y1;
        H = iNum * X2Y1 + iNum * Y3 - (X2 + Y2) * Y1;
        a = (H * D - E * G) / (C * G - D * D);
        b = (H * C - E * D) / (D * D - G * C);
        c = -(a * X1 + b * Y1 + X2 + Y2) / iNum;
        double A = 0.0;
        double B = 0.0;
        double R = 0.0;
        A = a / (-2);
        B = b / (-2);
        R = double(sqrt(a * a + b * b - 4 * c) / 2);
        R_center.x = A;
        R_center.y = B;
        Radius = R;
    }

    circle(src,R_center,Radius,Scalar(0,0,0),3);
    circle(src,R_center,4,Scalar(0,0,0),3);

    cout<<R_center<<endl<<Radius<<endl;
    imshow("circle",src);
}

void Windwill::Direction()
{
    if(Armor_Centers.size()==Params.infer_frame_number){

        Point2f dis1 = Point2f((Armor_Centers[Params.infer_frame_number1].x - R_center.x), (Armor_Centers[Params.infer_frame_number1].y - R_center.y));
        Point2f dis2 = Point2f((Armor_Centers[Params.infer_frame_number2].x - R_center.x), (Armor_Centers[Params.infer_frame_number2].y - R_center.y));

        double fai1 = atan(dis1.y / dis1.x);
        double fai2 = atan(dis2.y / dis2.x);

        if (fai1 > fai2)
            direction = CLKWISE;
        else
            direction = CCLKWISE;
    }

    cout<<(int)direction<<endl;
}

void Windwill::precise(){
    if(Armor_Centers.size()==Params.infer_frame_number){
        int n = Armor_Centers.size();

        Point2f temp;
        Point2f tt[4];
        for (int i = 0; i < n; i++)
        {
            Point2f points[4];
            temp += Armor_Centers[i];
            RotatedRect rect_temp = Armor_rect[i];
            rect_temp.points(points);

            tt[0]+=points[0];
            tt[1]+=points[1];
            tt[2]+=points[2];
            tt[3]+=points[3];
        }

        for(int i=0;i<4;i++){
            tt[i].x=tt[i].x/n;
            tt[i].y=tt[i].y/n;
        }

        points_2d_temp.push_back(tt[0]);
        points_2d_temp.push_back(tt[1]);
        points_2d_temp.push_back(tt[2]);
        points_2d_temp.push_back(tt[3]);

        Armor_temp_center.x = (temp.x) / n;
        Armor_temp_center.y = (temp.y) / n;
    }
}

Point2f Windwill::Min_Motion_Predict(Point2f point1)
{
    Point2f point2;

    float fai = atan((point1.y - R_center.y) / (point1.x - R_center.x));
    float tha = 2 * M_PI * Params.Min_n * shoot_time;

    if (direction == CLKWISE)
    {
        point2.x = Radius * cos(fai - tha) + R_center.x;
        point2.y = Radius * sin(fai - tha) + R_center.y;
    }
    else
    {
        point2.x = Radius * cos(fai + tha) + R_center.x;
        point2.y = Radius * sin(fai + tha) + R_center.y;
    }

    return point2;
}

Point2f Windwill::Max_Motion_Predict(Point2f point1)
{
    Point2f point2;

    double tha = 0;
    const int N = 50000;

    double delta = shoot_time / N;
    float fai = atan((point1.y - R_center.y) / (point1.x - R_center.x));

    for (double i = process_time + delta; i < process_time + shoot_time; i += delta)
    {
        tha += (double)funct(i + process_time) * delta;
    }

    if (direction == CLKWISE)
    {
        point2.x = Radius * cos(fai - tha) + R_center.x;
        point2.y = Radius * sin(fai - tha) + R_center.y;
    }
    else
    {
        point2.x = Radius * cos(fai + tha) + R_center.x;
        point2.y = Radius * sin(fai + tha) + R_center.y;
    }

    return point2;
}
