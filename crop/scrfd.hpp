#ifndef SCRFD_HPP
#define SCRFD_HPP

#include <opencv2/core/core.hpp>

#include "net.h"

struct FaceObject {
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

class SCRFD
{
    private:
        ncnn::Net scrfd;
        bool has_kps;
    public:
        int load(const char *model_type, const char *model_path);
        int detect(const cv::Mat &rgb, std::vector<FaceObject> &face_objects, float prob_threshold = 0.63f, float nms_threshold = 0.45f);
        int draw(cv::Mat &rgb, const std::vector<FaceObject> &face_objects);
};

#endif