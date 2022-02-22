#include "scrfd.hpp"

#include <iostream>
#include <string.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static inline float intersection_area(const FaceObject &a, const FaceObject &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<FaceObject> &face_objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = face_objects[(left + right) / 2].prob;

    while(i <= j) {
        while(face_objects[i].prob > p)
            i++;
        
        while(face_objects[j].prob < p)
            j--;

        if(i <= j) {
            std::swap(face_objects[i], face_objects[j]);
            i++;
            j--;
        }
    }

    {
        {
            if(left < j) qsort_descent_inplace(face_objects, left, j);
        }

        {
            if(i < right) qsort_descent_inplace(face_objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<FaceObject> &face_objects)
{
    if(face_objects.empty())
        return ;
    
    qsort_descent_inplace(face_objects, 0, face_objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<FaceObject> &face_objects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();

    const int n = face_objects.size();

    std::vector<float> areas(n);
    for(int i=0; i<n; i++) {
        areas[i] = face_objects[i].rect.area();
    }

    for(int i=0; i<n; i++) {
        const FaceObject &a = face_objects[i];

        int keep = 1;
        for(int j=0; j<(int)picked.size(); j++) {
            const FaceObject &b = face_objects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if(inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if(keep)
            picked.push_back(i);
    }
}

static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales)
{
    int num_ratio = ratios.w;
    int num_scale = scales.w;
	std::cout << num_ratio << std::endl;
	std::cout << num_scale << std::endl;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = 0;
    const float cy = 0;

    for(int i=0; i < num_ratio; i++) {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar);

        for(int j=0; j<num_scale; j++) {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float* anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }
    return anchors;
}

static void generate_proposals(const ncnn::Mat &anchors, int feat_stride, const ncnn::Mat &score_blob, const ncnn::Mat &bbox_blob, const ncnn::Mat &kps_blob, 
                                float prob_threshold, std::vector<FaceObject> &face_objects)
{
    int w = score_blob.w;
    int h = score_blob.h;
	std::cout << w << "x" << h << std::endl;

    const int num_anchors = anchors.h;

    for(int q=0; q<num_anchors; q++) {
        const float* anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);

        float anchor_y = anchor[1];
        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for(int i=0; i<h; i++) {
            float anchor_x = anchor[0];

            for(int j=0; j < w; j++) {
                int index = i * w + j;
                float prob = score[index];

                if(prob >= prob_threshold) {
                    float dx = bbox.channel(0)[index] * feat_stride;
                    float dy = bbox.channel(1)[index] * feat_stride;
                    float dw = bbox.channel(2)[index] * feat_stride;
                    float dh = bbox.channel(3)[index] * feat_stride;

                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float x0 = cx - dx;
                    float y0 = cy - dy;
                    float x1 = cx + dw;
                    float y1 = cy + dh;

                    FaceObject obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.prob = prob;

                    if(!kps_blob.empty()) {
                        const ncnn::Mat kps = kps_blob.channel_range(q * 10, 10);
                        obj.landmark[0].x = cx + kps.channel(0)[index] * feat_stride;
                        obj.landmark[0].y = cy + kps.channel(1)[index] * feat_stride;
                        obj.landmark[1].x = cx + kps.channel(2)[index] * feat_stride;
                        obj.landmark[1].y = cy + kps.channel(3)[index] * feat_stride;
                        obj.landmark[2].x = cx + kps.channel(4)[index] * feat_stride;
                        obj.landmark[2].y = cy + kps.channel(5)[index] * feat_stride;
                        obj.landmark[3].x = cx + kps.channel(6)[index] * feat_stride;
                        obj.landmark[3].y = cy + kps.channel(7)[index] * feat_stride;
                        obj.landmark[4].x = cx + kps.channel(8)[index] * feat_stride;
                        obj.landmark[4].y = cy + kps.channel(9)[index] * feat_stride;
                    }
                    face_objects.push_back(obj);
                }
                anchor_x += feat_stride;
            }
            anchor_y += feat_stride;
        }
    }
}

int SCRFD::load(const char* model_type, const char* model_path)
{
    scrfd.clear();
    
    char path_param[256];
    char path_bin[256];
    sprintf(path_param, "%s/scrfd_%s-opt2.param", model_path, model_type);
    sprintf(path_bin, "%s/scrfd_%s-opt2.bin", model_path, model_type);

    printf("param_path: %s\nbin_path: %s\n", path_param, path_bin);

    scrfd.load_param(path_param);
    scrfd.load_model(path_bin);

    has_kps = strstr(model_type, "_kps") != NULL;

    return 0;
}

int SCRFD::detect(const cv::Mat &rgb, std::vector<FaceObject> &face_objects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    const int target_size = 640;

    int w = width;
    int h = height;
    float scale = 1.f;
    if(w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float)target_size / h;
        h = target_size;
        h = h * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1/128.f, 1/128.f, 1/128.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = scrfd.create_extractor();

    ex.input("input.1", in_pad);

    std::vector<FaceObject> face_proposals;

    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_8", score_blob);
        ex.extract("bbox_8", bbox_blob);
        if(has_kps)
            ex.extract("kps_8", kps_blob);

std::cout << "score_blob: " << score_blob.w << "x" << score_blob.h << std::endl;
std::cout << "bbox_blob: " << bbox_blob.w << "x" << bbox_blob.h << std::endl;
std::cout << "kps_blob: " << kps_blob.w << "x" << kps_blob.h << std::endl;

        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);
std::cout << anchors.w << "x" << anchors.h << std::endl;

        std::vector<FaceObject> face_objects_32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, face_objects_32);

        face_proposals.insert(face_proposals.end(), face_objects_32.begin(), face_objects_32.end());
    }

    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_16", score_blob);
        ex.extract("bbox_16", bbox_blob);
        if(has_kps)
            ex.extract("kps_16", kps_blob);

        const int base_size = 64;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> face_objects_16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, face_objects_16);

        face_proposals.insert(face_proposals.end(), face_objects_16.begin(), face_objects_16.end());
    }

    // stride 32
    {
        ncnn::Mat score_blob, bbox_blob, kps_blob;
        ex.extract("score_32", score_blob);
        ex.extract("bbox_32", bbox_blob);
        if(has_kps)
            ex.extract("kps_32", kps_blob);

        const int base_size = 256;
        const int feat_stride = 32;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 1.f;
        scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<FaceObject> face_objects_8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, face_objects_8);

        face_proposals.insert(face_proposals.end(), face_objects_8.begin(), face_objects_8.end());
    }

    qsort_descent_inplace(face_proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(face_proposals, picked, nms_threshold);

    int face_count = picked.size();

    face_objects.resize(face_count);
    for(int i=0; i<face_count; i++) {
        face_objects[i] = face_proposals[picked[i]];

        float x0 = (face_objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (face_objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (face_objects[i].rect.x + face_objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (face_objects[i].rect.y + face_objects[i].rect.height - (hpad / 2)) / scale;

        x0 = std::max(std::min(x0, (float)width - 1), 0.f);
        y0 = std::max(std::min(y0, (float)height - 1), 0.f);
        x1 = std::max(std::min(x1, (float)width - 1), 0.f);
        y1 = std::max(std::min(y1, (float)height - 1), 0.f);

        face_objects[i].rect.x = x0;
        face_objects[i].rect.y = y0;
        face_objects[i].rect.width = x1 - x0;
        face_objects[i].rect.height = y1 - y0;

        if(has_kps) {
            float x0 = (face_objects[i].landmark[0].x - (wpad / 2)) / scale;
            float y0 = (face_objects[i].landmark[0].y - (hpad / 2)) / scale;
            float x1 = (face_objects[i].landmark[1].x - (wpad / 2)) / scale;
            float y1 = (face_objects[i].landmark[1].y - (hpad / 2)) / scale;
            float x2 = (face_objects[i].landmark[2].x - (wpad / 2)) / scale;
            float y2 = (face_objects[i].landmark[2].y - (hpad / 2)) / scale;
            float x3 = (face_objects[i].landmark[3].x - (wpad / 2)) / scale;
            float y3 = (face_objects[i].landmark[3].y - (hpad / 2)) / scale;
            float x4 = (face_objects[i].landmark[4].x - (wpad / 2)) / scale;
            float y4 = (face_objects[i].landmark[4].y - (hpad / 2)) / scale;

            face_objects[i].landmark[0].x = std::max(std::min(x0, (float)width - 1), 0.f);
            face_objects[i].landmark[0].y = std::max(std::min(y0, (float)height - 1), 0.f);
            face_objects[i].landmark[1].x = std::max(std::min(x1, (float)width - 1), 0.f);
            face_objects[i].landmark[1].y = std::max(std::min(y1, (float)height - 1), 0.f);
            face_objects[i].landmark[2].x = std::max(std::min(x2, (float)width - 1), 0.f);
            face_objects[i].landmark[2].y = std::max(std::min(y2, (float)height - 1), 0.f);
            face_objects[i].landmark[3].x = std::max(std::min(x3, (float)width - 1), 0.f);
            face_objects[i].landmark[3].y = std::max(std::min(y3, (float)height - 1), 0.f);
            face_objects[i].landmark[4].x = std::max(std::min(x4, (float)width - 1), 0.f);
            face_objects[i].landmark[4].y = std::max(std::min(y4, (float)height - 1), 0.f);
        }
    }

    return 0;
}

int SCRFD::draw(cv::Mat &rgb, const std::vector<FaceObject> &face_objects)
{
    for(size_t i=0; i<face_objects.size(); i++) {
        const FaceObject &obj = face_objects[i];

        cv::rectangle(rgb, obj.rect, cv::Scalar(0, 255, 0), 2);

        if(has_kps) {
            cv::circle(rgb, obj.landmark[0], 2, cv::Scalar(0, 255, 0), 2);
            cv::circle(rgb, obj.landmark[1], 2, cv::Scalar(0, 255, 0), 2);
            cv::circle(rgb, obj.landmark[2], 2, cv::Scalar(0, 255, 0), 2);
            cv::circle(rgb, obj.landmark[3], 2, cv::Scalar(0, 255, 0), 2);
            cv::circle(rgb, obj.landmark[4], 2, cv::Scalar(0, 255, 0), 2);
        }

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if(y < 0)
            y = 0;
        if(x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(0, 255, 0), 1);
        
        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    return 0;
}
