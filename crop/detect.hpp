
#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP

#include <string>
#include <stack>

#include "opencv2/opencv.hpp"

#include "net.h"

struct ToaDoLandmark {
    float _x;
    float _y;
};

struct bbox {
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    ToaDoLandmark point[5];
};

struct box {
    float cx;
    float cy;
    float sx;
    float sy;
};

class FaceDetect 
{
    private:

    public:
        FaceDetect();

        int Init(const std::string &model_path);

        inline void Release();

        void nms(std::vector <bbox> &input_boxes, float NMS_THRESH);

        void Detect(cv::Mat &img, std::vector <bbox> &boxes);

        void create_anchor(std::vector <box> &anchor, int w, int h);

        void create_anchor_retinaface(std::vector <box> &anchor, int w, int h);

        inline void SetDefaultParams();

        static inline bool cmp(bbox a, bbox b);

        ~FaceDetect();

    public:
        float _nms;
        float _threshold;
        float _mean_val[3];
        bool _retinaface;

        ncnn::Net *Net;
};

#endif //