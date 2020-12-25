#include "windmill.h"

#define WIND_DEBUG

#ifdef WIND_DEBUG
//#define SINGAL
//#define SHOW_COLOR_TEMP_OTSU
//#define SHOW_AIM_SINGLE_CIRCLE
//#define SHOW_TIME_ISFIND
//#define SHOW_FARTHER
//#define SHOW_DIRECTION
#define SHOW_PRECENTER
#endif

WindMill::WindMill(){
}

/// \brief tracking
void WindMill::Tracking(const Mat track, Rect2d rect){
    tracker->update(src,rect);
    rectangle(track,rect,Scalar(0,0,255),3);
    imshow("tracking",track);
}


/// \brief 清空数据
void WindMill::clear(){
    detect_mode = 0;
    frame_cnt = 0;
    memset((void*)&lastData,0x00,sizeof(armorData));
    memset((void*)&lostData,0x00,sizeof(armorData));
}

/// \brief 保证用 rect 截图安全
/// \param rect 图中的roi范围
/// \param size 图的大小
bool WindMill::makeRectSafe(const Rect rect,const Size size){
    if (rect.x < 0)
        return false;
    if (rect.x + rect.width > size.width)
        return false;
    if (rect.y < 0)
        return false;
    if (rect.y + rect.height > size.height)
        return false;
    if (rect.width <= 0 || rect.height <= 0)
        return false;
    return true;
}

/// \brief 根据点集使用最小二乘法拟合圆
/// \param points 点集
/// \param R_center 圆心
bool WindMill::circleLeastFit(const vector<Point2f> &points,Point2f &R_center){
    float center_x = 0.0f;
    float center_y = 0.0f;
    //float radius = 0.0f;

    if (points.size() < 3){
        return false;
    }

    double sum_x = 0.0f, sum_y = 0.0f;
    double sum_x2 = 0.0f, sum_y2 = 0.0f;
    double sum_x3 = 0.0f, sum_y3 = 0.0f;
    double sum_xy = 0.0f, sum_x1y2 = 0.0f, sum_x2y1 = 0.0f;

    int N = points.size();
    for (int i = 0; i < N; i++){
        double x = points[i].x;
        double y = points[i].y;
        double x2 = x * x;
        double y2 = y * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x2;
        sum_y2 += y2;
        sum_x3 += x2 * x;
        sum_y3 += y2 * y;
        sum_xy += x * y;
        sum_x1y2 += x * y2;
        sum_x2y1 += x2 * y;
    }

    double C, D, E, G, H;
    double a, b;//c;

    C = N * sum_x2 - sum_x * sum_x;
    D = N * sum_xy - sum_x * sum_y;
    E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x;
    G = N * sum_y2 - sum_y * sum_y;
    H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y;
    a = (H * D - E * G) / (C * G - D * D);
    b = (H * C - E * D) / (D * D - G * C);
    //c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N;

    center_x = a / (-2);
    center_y = b / (-2);
    //radius = sqrt(a * a + b * b - 4 * c) / 2;
    R_center = Point2f(center_x,center_y);

#ifdef SHOW_CIRCLE
    circle(src,R_center,radius,Scalar(255,0,0),3);
    circle(src,R_center,4,Scalar(255,0,0),3);

    //    cout<<R_center<<endl<<radius<<endl;
    imshow("circle",src);
#endif

    return true;
}

/// \brief 检测装甲板
/// \param src 原图
/// \param data 装甲板信息
bool WindMill::getArmorCenter(const Mat src,armorData &data){

    /********************************* 预处理 ************************************/

    mat_track=src.clone();

    clock_t finish;
    start = clock();

    vector<Mat> channels;
    Mat color,temp,temp1,temp2,Leaf_preImg;
    split(src, channels);

    if (param.Enemy_color == RED)
        color = channels[2] - channels[0];
    else
        color = channels[0] - channels[2];

    GaussianBlur(color,color,Size(3,3),1);

    threshold(color, temp1, param.max_imgthresold, 255, THRESH_BINARY);
    threshold(color, temp2, param.min_imgthresold, 255, THRESH_BINARY);

    add(temp1,temp2,temp);

    int elementSize = 3;
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * elementSize + 1, 2 * elementSize + 1),Size(elementSize,elementSize));
    dilate(temp, temp, element);

    floodFill(temp,Point(0,0),Scalar(0));

    elementSize = 2;
    element = getStructuringElement(MORPH_RECT, Size(2 * elementSize + 1, 2 * elementSize + 1),Size(elementSize,elementSize));
    morphologyEx(temp, temp, MORPH_CLOSE, element);

    medianBlur(color,color,5);
    Leaf_preImg = temp.clone();

#ifdef SHOW_COLOR_TEMP_OTSU
    //imshow("color",color);
    //imshow("temp1",temp1);
    //imshow("temp2",temp2);
    imshow("OTSU",temp);
#ifdef SINGAL
    waitKey(0);
#endif
#endif

    /******************************* 检测目标 *************************************/
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(Leaf_preImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    if (!hierarchy.empty() && !contours.empty())
    {
        for (int i = 0; (i>=0) && (hierarchy[i][3]< 0); i=hierarchy[i][0])
        {
            RotatedRect leaf_temp_rect = minAreaRect(contours[i]);

            Point2f p[4];
            leaf_temp_rect.points(p);

            Point2f srcRect[4];
            Point2f dstRect[4];

            double width = distance(p[0], p[1]);
            double height = distance(p[1], p[2]);

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
            if (contourArea(contours[i])> param.Leaf_minArea)
            {

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

                if(svm->predict(svm_mat) >= 0.95 && (hierarchy[i][2] > 0))
                {
                    if (hierarchy[hierarchy[i][2]][1] < 0 && hierarchy[hierarchy[i][2]][0] < 0)
                    {
#ifdef SHOW_FARTHER
                        Mat armor1=Mat(src,temp1);
                        imshow("father",armor1);
#ifdef SINGAL
                        waitKey(0);
#endif
#endif
                        Point2f arrowCenter = leaf_temp_rect.center;

                        RotatedRect armor_rect = minAreaRect(contours[hierarchy[i][2]]);
                        Rect rect=boundingRect(contours[hierarchy[i][2]]);

                        static bool track_init_once = true;

//                        if(track_init_once)
//                            aim_rect = (Rect2d)rect;
//                            tracker->init(mat_track,aim_rect);
                        track_init_once = false;

                        Point2f pnt[4];
                        armor_rect.points(pnt);

                        float width = armor_rect.size.width;
                        float height = armor_rect.size.height;

                        if (height > width)
                            swap(width, height);

                        if ((height / width) > param.Armor_maxHWRation)
                            continue;

                        if(!makeRectSafe(rect,src.size())) return false;

                        data.isFind = true;
                        data.armorCenter = armor_rect.center;
                        data.armorRect = armor_rect;
                        data.angle = armor_rect.angle;

                        double k=3.215321;
                        data.RCenter = (1-k)*armor_rect.center+k*arrowCenter;
                        data.Radius=distance(armor_rect.center,data.RCenter);

                        for(int i=0;i<4;i++)
                            data.Rect_Radius[i]=distance(pnt[i],data.RCenter);

#ifdef SHOW_AIM_SINGLE_CIRCLE
                        for (int i = 0; i < 4; i++)
                            line(src, p[i], p[(i+1)%4], Scalar(0,0,255),2);

                        for (int i = 0; i < 4; i++)
                            line(src, pnt[i], pnt[(i+1)%4], Scalar(0,0,255),3);

                        circle(src,armor_rect.center,3,Scalar(0,0,255),3,LINE_8,0);
                        circle(src,arrowCenter,3,Scalar(0,0,255),3,LINE_8,0);
                        line(src,armor_rect.center,data.RCenter,Scalar(0,0,255),2);
                        circle(src,data.RCenter,3,Scalar(0,0,255),3,LINE_8,0);

                        imshow("aim",src);
#ifdef SINGAL

                        waitKey(0);
#endif
#endif

                        float tran_angle = 0.0;
                        if(armor_rect.size.width > armor_rect.size.height){
                            tran_angle = 90 - fabs(armor_rect.angle);
                        }else{
                            tran_angle = fabs(armor_rect.angle);
                        }

                        if(tran_angle < 20){
                            if(armor_rect.size.width < armor_rect.size.height
                                    && arrowCenter.x < data.armorCenter.x){
                                data.quadrant = 1;
                            }else if(armor_rect.size.width > armor_rect.size.height
                                     && arrowCenter.x > data.armorCenter.x){
                                data.quadrant = 2;
                            }else if(armor_rect.size.width < armor_rect.size.height
                                     && arrowCenter.x > data.armorCenter.x){
                                data.quadrant = 3;
                            }else if(armor_rect.size.width > armor_rect.size.height
                                     && arrowCenter.x < data.armorCenter.x){
                                data.quadrant = 4;
                            }
                        }else if(tran_angle > 70){
                            if(armor_rect.size.width < armor_rect.size.height
                                    && arrowCenter.y > data.armorCenter.y){
                                data.quadrant = 1;
                            }else if(armor_rect.size.width > armor_rect.size.height
                                     && arrowCenter.y > data.armorCenter.y){
                                data.quadrant = 2;
                            }else if(armor_rect.size.width < armor_rect.size.height
                                     && arrowCenter.y < data.armorCenter.y){
                                data.quadrant = 3;
                            }else if(armor_rect.size.width > armor_rect.size.height
                                     && arrowCenter.y < data.armorCenter.y){
                                data.quadrant = 4;
                            }
                        }else{
                            if(arrowCenter.x < data.armorCenter.x && arrowCenter.y >= data.armorCenter.y
                                    && armor_rect.size.width < armor_rect.size.height){
                                data.quadrant = 1;
                            }else if(arrowCenter.x >= data.armorCenter.x && arrowCenter.y > data.armorCenter.y
                                     && armor_rect.size.width >= armor_rect.size.height){
                                data.quadrant = 2;
                            }else if(arrowCenter.x > data.armorCenter.x && arrowCenter.y <= data.armorCenter.y
                                     && armor_rect.size.width < armor_rect.size.height){
                                data.quadrant = 3;
                            }else if(arrowCenter.x <= data.armorCenter.x && arrowCenter.y < data.armorCenter.y
                                     && armor_rect.size.width >= armor_rect.size.height){
                                data.quadrant = 4;
                            }
                        }

                        finish=clock();
                        data.time=(double)(finish-start)/CLOCKS_PER_SEC;

                    }
                }
            }
        }
    }
#ifdef SHOW_TIME_ISFIND
    cout<<data.isFind<<endl;
    cout<<data.time <<endl;
#endif
    return data.isFind;

}

/// \brief 返回SVM_MAT
/// \param input HOG处理后的MAT
Mat WindMill::getSvmInput(Mat &input)
{
    vector<float> vec=stander(input);
    Mat output(1,900,CV_32FC1);

    Mat_<float> p=output;
    int jj=0;
    for(vector<float>::iterator iter=vec.begin();iter!=vec.end();iter++,jj++)
    {
        p(0,jj)=*(iter);
    }
    return output;
}

/// \brief HOG处理
/// \param im 待判断二值化图片
vector<float> WindMill::stander(Mat &im)
{
    if(im.empty()==1)
        cout<<"IMG SVM can't convert"<<endl;
    resize(im,im,Size(48,48));
    vector<float> result;
    HOGDescriptor hog(Size(48,48),Size(16,16),Size(8,8),Size(8,8),9,1,-1,
                      HOGDescriptor::L2Hys,0.2,false,HOGDescriptor::DEFAULT_NLEVELS);
    hog.compute(im,result);
    return result;
}

/// \brief 判断顺逆时针旋转
bool WindMill::getDirection(){
    param.infer_frame_number = (shoot_mode==SHOOT_MIN)? 3:30;

    static vector<armorData> datas;
    static int times = 0;

    if(times < param.infer_frame_number){
        datas.push_back(lastData);
        times++;
        return false;
    }
    else{
        if(int(datas.size()) != param.infer_frame_number) {
            times = 0;
            datas.clear();
            return false;
        }

        float angles[param.infer_frame_number];
        memset(angles,0x00,sizeof(angles));
        for(int i=0;i<param.infer_frame_number;i++){
            float tran_angle = 0.0;
            if(datas[i].armorRect.size.width> datas[i].armorRect.size.height){
                tran_angle = 90 - fabs(datas[i].angle);
            }else{
                tran_angle = fabs(datas[i].angle);
            }

            change_angle(datas[i].quadrant, tran_angle, angles[i]);
        }

        int positive = 0;
        int negetive = 0;
        for(int j=1;j<3;++j){
            for(int i=0;i<param.infer_frame_number-j;++i){
                if((angles[i] - angles[i+j]) > 0 || (angles[i] - angles[i+j]) < -300){
                    positive++;
                }else if((angles[i] - angles[i+j]) < 0 || (angles[i] - angles[i+j]) > 300){
                    negetive++;
                }
            }
        }

        if(positive > negetive){
            if(param.Enemy_color==RED) detect_mode=RED_CLOCK;
            else detect_mode=RED_CLOCK;

#ifdef SHOW_DIRECTION
            cout<<"顺时针"<<endl;
#endif
        }else if(positive < negetive){
            if(param.Enemy_color==BLUE) detect_mode=BLUE_ANCLOCK;
            else detect_mode=BLUE_ANCLOCK;
#ifdef SHOW_DIRECTION
            cout<<"逆时针"<<endl;
#endif
        }
        times = 0;
        vector<armorData>().swap(datas);
        return true;
    }
}

/// \brief 根据象限把旋转矩形的角度转为360
/// \param quadrant 象限
/// \param angle 原始角度
/// \param tran_angle 转化的角度
bool WindMill::change_angle(const int quadrant, const float angle, float &tran_angle){
    if(quadrant == 1){
        tran_angle = angle;
    }else if(quadrant == 2){
        tran_angle = 90  +  90 - angle;
    }else if(quadrant == 3){
        tran_angle = 180 + angle;
    }else if(quadrant == 4){
        tran_angle = 270 + 90  - angle;
    }else{
        //cout << "象限为0" << endl;
        return false;
    }
    return true;
}

/// \brief 预测
/// \param data 装甲板信息
/// \param preCenter 预测点
/// \param pMode 预测方法
bool WindMill::predict(const armorData data,Point2f& preCenter,int pMode)
{
    // 根据点的集合拟合圆

    if(detect_mode==RED_STATIC || detect_mode==BLUE_STATIC)
        return false;

    Point2f pnts[4];
    data.armorRect.points(pnts);

    float preAngle;
    preAngle=(shoot_mode==SHOOT_MIN)? CV_PI/8 : Max_Motion_Predict(data);

    if(Max_Motion_Predict(data) == 0.0 && shoot_mode == SHOOT_MAX) return false;

    if(pMode == FIT_CIRCLE)
    {
        static int count = 0;
        static vector<Point2f> armorPoints;
        if(count < param.infer_frame_number){
            armorPoints.push_back(data.armorCenter);
            count++;
            return false;
        }
        else if(count == param.infer_frame_number){
            Point2f center;
            if (!circleLeastFit(armorPoints,center)) return false;

            if(detect_mode == RED_ANCLOCK || detect_mode == BLUE_ANCLOCK){
                preAngle = preAngle;
            }else{
                preAngle = -preAngle;
            }
            double x = data.armorCenter.x - center.x;
            double y = data.armorCenter.y - center.y;
            preCenter.x = x * cos(preAngle) + y * sin(preAngle) + center.x;
            preCenter.y = -x * sin(preAngle) + y * cos(preAngle) + center.y;

            for(int i=0;i<4;i++){
                Point2f temp;
                double x = pnts[i].x - center.x;
                double y = pnts[i].y - center.y;
                temp.x = x * cos(preAngle) + y * sin(preAngle) + center.x;
                temp.y = -x * sin(preAngle) + y * cos(preAngle) + center.y;
                pre_points[i]=temp;
            }
            return true;
        }
    }
    else if(pMode == PUSH_CIRCLE)
    {
        if(data.RCenter == Point2f(0,0)){
            return false;
        }

        if(detect_mode == RED_ANCLOCK || detect_mode == BLUE_ANCLOCK){
            preAngle = preAngle;
        }else{
            preAngle = -preAngle;
        }

        double x = data.armorCenter.x - data.RCenter.x;
        double y = data.armorCenter.y - data.RCenter.y;
        preCenter.x = x * cos(preAngle) + y * sin(preAngle) + data.RCenter.x;
        preCenter.y = -x * sin(preAngle) + y * cos(preAngle) + data.RCenter.y;

        for(int i=0;i<4;i++){
            Point2f temp;
            double x = pnts[i].x - data.RCenter.x;
            double y = pnts[i].y - data.RCenter.y;
            temp.x = x * cos(preAngle) + y * sin(preAngle) + data.RCenter.x;
            temp.y = -x * sin(preAngle) + y * cos(preAngle) + data.RCenter.y;
            pre_points[i]=temp;
        }
        return true;

    }
    else if(pMode == TANGENT)
    {// 先预测到切线方向，然后在切线方向上进行旋转的补偿

        float dis = data.Radius * tan(preAngle);
        float dis_x = dis * sin(data.angle * CV_PI/180);
        float dis_y = dis * cos(data.angle * CV_PI/180);

        Point2f tangent;
        if(detect_mode == RED_ANCLOCK || detect_mode == BLUE_ANCLOCK)
        {// 逆时针
            // 分四个象限
            if(data.quadrant == 1){
                tangent.x = data.armorCenter.x - dis_x;
                tangent.y = data.armorCenter.y - dis_y;
            }else if(data.quadrant == 2){
                tangent.x = data.armorCenter.x - dis_x;
                tangent.y = data.armorCenter.y + dis_y;
            }else if(data.quadrant == 3){
                tangent.x = data.armorCenter.x + dis_x;
                tangent.y = data.armorCenter.y + dis_y;
            }else if(data.quadrant == 4){
                tangent.x = data.armorCenter.x + dis_x;
                tangent.y = data.armorCenter.y - dis_y;
            }else{
                return false;
            }
            // 绕装甲板旋转
            double x =  tangent.x - data.armorCenter.x;
            double y =  tangent.y - data.armorCenter.y;
            preCenter.x = x * cos(preAngle/2) + y * sin(preAngle/2) + data.armorCenter.x;
            preCenter.y = -x * sin(preAngle/2) + y * cos(preAngle/2) + data.armorCenter.y;

            for(int i=0;i<4;i++){
                Point2f temp;
                double x = tangent.x - pnts[i].x;
                double y = tangent.y - pnts[i].y;
                temp.x = x * cos(preAngle/2) + y * sin(preAngle/2) + pnts[i].x;
                temp.y = -x * sin(preAngle) + y * cos(preAngle) + pnts[i].y;
                pre_points[i]=temp;
            }

        }
        else{
            if(data.quadrant == 1){
                tangent.x = data.armorCenter.x + dis_x;
                tangent.y = data.armorCenter.y + dis_y;
            }else if(data.quadrant == 2){
                tangent.x = data.armorCenter.x + dis_x;
                tangent.y = data.armorCenter.y - dis_y;
            }else if(data.quadrant == 3){
                tangent.x = data.armorCenter.x - dis_x;
                tangent.y = data.armorCenter.y - dis_y;
            }else if(data.quadrant == 4){
                tangent.x = data.armorCenter.x - dis_x;
                tangent.y = data.armorCenter.y + dis_y;
            }else{
                return false;
            }
            // 绕装甲板旋转
            double x =  tangent.x - data.armorCenter.x;
            double y =  tangent.y - data.armorCenter.y;
            preCenter.x = x * cos(-preAngle/2) + y * sin(-preAngle/2) + data.armorCenter.x;
            preCenter.y = -x * sin(-preAngle/2) + y * cos(-preAngle/2) + data.armorCenter.y;

            for(int i=0;i<4;i++){
                Point2f temp;
                double x = tangent.x - pnts[i].x;
                double y = tangent.y - pnts[i].y;
                temp.x = x * cos(preAngle/2) + y * sin(preAngle/2) + pnts[i].x;
                temp.y = -x * sin(preAngle) + y * cos(preAngle) + pnts[i].y;
                pre_points[i]=temp;
            }
        }

        return true;
    }

}
/// \brief 大符preAngle
double WindMill::Max_Motion_Predict(const armorData new_data)
{
    double tha = 0;
    const int N = 50000;
    double delta = param.shoot_time / N;

    static vector<double> ws;
    static bool _peak = false;
    static bool _valley = false;
    //static int times = 0;

    double w=distance(new_data.armorCenter,lastData.armorCenter)/
            ((new_data.Radius+lastData.Radius)/2)/new_data.time;
    ws.push_back(w);

    if(ws.size()<3)  return 0.0;

    if(ws[ws.size()-1]>w && ws[ws.size()-2]<ws[ws.size()-1]){
        _peak = true;
        //times = ws.size();
    }
    else if(ws[ws.size()-1]<w && ws[ws.size()-2]>ws[ws.size()-1]){
        _valley = true;
        //times = ws.size();
    }
    else return 0.0;

    if(_peak){
        for (double i = new_data.time; i < param.shoot_time + new_data.time; i += delta)
            tha += Max_W_Function(i,CV_PI/2) * delta;
    }
    if(_valley){
        for (double i = new_data.time; i < param.shoot_time + new_data.time; i += delta)
            tha += Max_W_Function(i,-CV_PI/2) * delta;
    }

    vector<double>().swap(ws);
    _peak = false;
    _valley = false;
    //times = 0;

    return tha;
}

/// \brief 大符Function
double WindMill::Max_W_Function(double t,double fai){
    return 0.785*sin(1.884*t+fai)+1.305;//rad/s
}

/// \brief 判断是否切换
/// \param new_data 新的数据
/// \param status 当前的状态
void WindMill::isCut(const armorData new_data,int &status){

    // 连续两帧都识别到
    if(new_data.isFind == true && lastData.isFind == true){

        // 新数据角度
        float new_tran_angle = 0.0;
        change_angle(new_data.quadrant,new_data.angle,new_tran_angle);

        // 旧数据角度
        float last_tran_angle = 0.0;
        change_angle(lastData.quadrant,lastData.angle,last_tran_angle);

        // 1和4象限边界
        if(new_data.quadrant == 4 && lastData.quadrant == 1){
            last_tran_angle += 360;
        }else if(new_data.quadrant == 1 && lastData.quadrant == 4){
            new_tran_angle += 360;
        }

        float dis = fabs(new_tran_angle - last_tran_angle);
        if(dis < 40){
            status = 1;
        }else{
            status = 2;
            // 对切换指令作限制
            if((int)frame_cnt < param.cutLimitedTime){// 400ms
                status = 1;
            }else{
                frame_cnt = 0;
            }
        }
    }

    // 掉帧后开始识别到
    else if(new_data.isFind == true
            && lastData.isFind == false){

        if(lostData.isFind == true){
            // 新数据角度
            float new_tran_angle = 0.0;
            change_angle(new_data.quadrant,new_data.angle,new_tran_angle);

            // 最后一次丢失的角度
            float lost_tran_angle = 0.0;
            change_angle(lostData.quadrant,lostData.angle,lost_tran_angle);

            // 1和4象限边界
            if(new_data.quadrant == 4 && lostData.quadrant == 1){
                lost_tran_angle += 360;
            }else if(new_data.quadrant == 1 && lostData.quadrant == 4){
                new_tran_angle += 360;
            }

            float dis = fabs(new_tran_angle - lost_tran_angle);
            if(dis < 50){
                status = 1;
            }else{
                status = 2;
                // 对切换指令作限制
                if((int)frame_cnt < param.cutLimitedTime){// 400ms
                    status = 1;
                }else{
                    frame_cnt = 0;
                }
            }
        }
        else if(lostData.isFind == false){
            status = 2; // 第一帧开始识别,直接给出切换
        }
    }
    // 第一帧开始掉，记录最后一次的数据
    else if(new_data.isFind == false && lastData.isFind == true){
        lostData = lastData;
        status = 0;
    }
    // 一直识别不到
    else if(new_data.isFind == false && lastData.isFind == false){
        status = 0;
    }
    frame_cnt++;
}

/// \brief 检测函数
/// \param frame 图
/// \param Mode 接受到的指令
/// \param pt,pts 返回（预测后）Armor的中心坐标,装甲板坐标
/// \param rect 返回 minAreaRect(pts)后的斜矩形（测试）
/// \param status 返回当前的状态
void WindMill::detect(const Mat frame,int Mode,Point2f &pt, RotatedRect &rect, Point2f pts[],int &status){

    shoot_mode = Mode;
    src = frame;
    vector<Point2f> p;

    // Detect the armor
    if(shoot_mode==SHOOT_MIN)
    {
        armorData armordata;
        param.pMode = PUSH_CIRCLE;

        if(getArmorCenter(src,armordata) == false){
            pt = Point2f(0,0);
            for(int i=0;i<4;i++)
                pts[i] = Point2f(0,0);
        }
        else{
            Point2f preCenter;
            if(predict(armordata,preCenter,param.pMode) == false){
                pt = Point2f(0,0);
                for(int i=0;i<4;i++)
                    pts[i] = Point2f(0,0);
            }

            else{
                if(preCenter!=Point2f(0,0))
                    pt = preCenter;
                for(int i=0;i<4 && pre_points[0]!=Point2f(0,0);i++){
                    p.push_back(pre_points[i]);
                    pts[i] = pre_points[i];
                }
                rect = minAreaRect(p);
                vector<Point2f>().swap(p);
#ifdef SHOW_PRECENTER
                for (int i = 0; i < 4; i++)
                    line(src, pre_points[i], pre_points[(i+1)%4], Scalar(0,255,0),3);
                circle(src,preCenter,3,Scalar(0,0,255),2);
                imshow("precenter",src);
#ifdef SINGAL
#endif
#endif
                if(detect_mode==RED_ANCLOCK || detect_mode==RED_CLOCK)
                    detect_mode=RED_STATIC;
                else
                    detect_mode=BLUE_STATIC;
            }
        }
        getDirection();
        isCut(armordata,status);
        lastData=armordata;

        if(status== 0 && pt != Point2f(0,0)){
            status = 1;
        }
    }
    else if(shoot_mode==SHOOT_MAX)
    {
        armorData armordata;
        param.pMode = FIT_CIRCLE;

        if(getArmorCenter(src,armordata) == false){
            pt = Point2f(0,0);
            for(int i=0;i<4;i++)
                pts[i] = Point2f(0,0);
        }
        else{
            Point2f preCenter;
            if(predict(armordata,preCenter,param.pMode) == false){
                pt = Point2f(0,0);
                for(int i=0;i<4;i++)
                    pts[i] = Point2f(0,0);
            }
            else{
                if(preCenter!=Point2f(0,0))
                    pt = preCenter;
                for(int i=0;i<4 && pre_points[0]!=Point2f(0,0);i++){
                    p.push_back(pre_points[i]);
                    pts[i] = pre_points[i];
                }
                rect = minAreaRect(p);
                vector<Point2f>().swap(p);
#ifdef SHOW_PRECENTER
                for (int i = 0; i < 4; i++)
                    line(src, pre_points[i], pre_points[(i+1)%4], Scalar(255,255,0),3);
                circle(src,preCenter,5,Scalar(0,0,0),2);
                imshow("precenter",src);
#ifdef SINGAL
#endif
#endif
                if(detect_mode==RED_ANCLOCK || detect_mode==RED_CLOCK)
                    detect_mode=RED_STATIC;
                else
                    detect_mode=BLUE_STATIC;
            }
        }
        getDirection();
        isCut(armordata,status);
        lastData=armordata;

        if(status== 0 && pt != Point2f(0,0)){
            status = 1;
        }
    }
    else{
        return;
    }
}
