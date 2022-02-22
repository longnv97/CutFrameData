// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"
#include <glob.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/opencv.hpp>
#if CV_MAJOR_VERSION >= 3
#include <opencv2/videoio/videoio.hpp>
#endif
#include <string.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <bits/stdc++.h> 
#include <fstream>
#include <chrono>

#include "detect.hpp"
#include "scrfd.hpp"

#define NCNN_PROFILING
#define YOLOV4_TINY //Using yolov4_tiny, if undef, using original yolov4
// using namespace std;
// using namespace cv;
#ifdef NCNN_PROFILING
#include "benchmark.h"
#endif
int soluong = 0;
static const char* class_names[] = { "background", "0",
"1",
"2",
"3",
"4",
"5",
"6",
"7",
"8",
"9",
"A",
"B",
"C",
"D",
"E",
"F",
"G",
"H",
"I",
"J",
"K",
"L",
"M",
"N",
"O",
"P",
"Q",
"R",
"S",
"T",
"U",
"V",
"W",
"X",
"Y",
"Z"
    };

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    cv::Mat image;
};

std::vector<FaceObject> face_objects;
SCRFD scrfd;
const char* model_type = "500m_kps";
int target_size = 512;
cv::Mat frame;
std::vector<Object> objects;
ncnn::Net yolov4;
ncnn::Net classifier;
std::string pathToParam = "/home/longnve/AICam/CutFrameData/crop/humandetect.param";
std::string pathToBin = "/home/longnve/AICam/CutFrameData/crop/humandetect.bin";
std::string pathToFolderImage = "/media/longnve/Data/DataAI/DataBphone/10.2.21.228/input";
std::string pathToFolderResult = "/media/longnve/Data/DataAI/DataBphone/10.2.21.228/output";

// std::string pathToFolderImage = "/media/tienvh/BkavSoft/1.Work/FaceMask/HH1_image/cropped";
// std::string pathToFolderResult = "/media/tienvh/BkavSoft/1.Work/FaceMask/HH1_image/cropped_result";

static int init_yolov4(ncnn::Net* yolov4, const char* param, const char* model, int target_size)
{
    /* --> Set the params you need for the ncnn inference <-- */

    yolov4->opt.num_threads = 4; //You need to compile with libgomp for multi thread support

    yolov4->opt.use_vulkan_compute = true; //You need to compile with libvulkan for gpu support

    yolov4->opt.use_winograd_convolution = true;
    yolov4->opt.use_sgemm_convolution = true;
    yolov4->opt.use_fp16_packed = true;
    yolov4->opt.use_fp16_storage = true;
    yolov4->opt.use_fp16_arithmetic = true;
    yolov4->opt.use_packing_layout = true;
    yolov4->opt.use_shader_pack8 = false;
    yolov4->opt.use_image_storage = false;

    /* --> End of setting params <-- */
    int ret = 0;

    // original pretrained model from https://github.com/AlexeyAB/darknet
    // the ncnn model https://drive.google.com/drive/folders/1YzILvh0SKQPS_lrb33dmGNq7aVTKPWS0?usp=sharing
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
#ifdef YOLOV4_TINY
    const char* yolov4_param = param;
    const char* yolov4_model = model;
    // *target_size = 608;
#else
    const char* yolov4_param = "yolov4-opt.param";
    const char* yolov4_model = "yolov4-opt.bin";
    // *target_size = 608;
#endif

    ret = yolov4->load_param(yolov4_param);
    if (ret != 0)
    {
        return ret;
    }

    ret = yolov4->load_model(yolov4_model);
    if (ret != 0)
    {
        return ret;
    }

    return 0;
}

static int detect_yolov4(const cv::Mat& frame, std::vector<Object>& objects, int target_size, ncnn::Net* yolov4,  std::string name)
{
    int ret = -1;
    int img_w = frame.cols;
    int img_h = frame.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows, target_size, target_size);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov4->create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);

    objects.clear();

    std::cout << "image name: " << name << std::endl;
    std::cout << "number face: " << out.h << std::endl;
	
    for (int i = 0; i < out.h; i++)
    {
        std::cout << "process object ---" << std::endl;
        const float* values = out.row(i);

        Object object;
        int padding = 40;

        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        if (object.prob > 0.9)
        {
            // cv::rectangle(frame, object.rect, cv::Scalar(41, 0, 223), 2 , 3);
            // cv::putText(frame, class_names[(int)values[0]] + std::to_string(object.prob), cv::Point(object.rect.x, object.rect.y + object.rect.height / 2),
            // cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);

            objects.push_back(object);
        }
    }
    if (objects.size() != 0){
            soluong= soluong +1;
            cv::imwrite(name, frame);
        }
    // std::cout<< " So luong : "<< soluong<< std::endl;
	// cv::imwrite(name, frame);
    return 0;
}

static int yolov4_detect_cropped_image(const cv::Mat& cropped_image, Object &object, int target_size, ncnn::Net* yolov4)
{
    object.label = 0;
    object.prob = 0;
    int img_w = cropped_image.cols;
    int img_h = cropped_image.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(cropped_image.data, ncnn::Mat::PIXEL_BGR2RGB, cropped_image.cols, cropped_image.rows, target_size, target_size);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov4->create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);

    std::cout << "number face: " << out.h << std::endl;

    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);
        object.label = values[0];
        object.prob = values[1];
        if (object.prob < 0.6)
            object.label = 0;
    }

    return 0;
}

static int init_classification_model(ncnn::Net* classifier, const char* param, const char* bin)
{
    /* --> Set the params you need for the ncnn inference <-- */

    classifier->opt.num_threads = 4; //You need to compile with libgomp for multi thread support

    classifier->opt.use_vulkan_compute = true; //You need to compile with libvulkan for gpu support

    classifier->opt.use_winograd_convolution = true;
    classifier->opt.use_sgemm_convolution = true;
    classifier->opt.use_fp16_packed = true;
    classifier->opt.use_fp16_storage = true;
    classifier->opt.use_fp16_arithmetic = true;
    classifier->opt.use_packing_layout = true;
    classifier->opt.use_shader_pack8 = false;
    classifier->opt.use_image_storage = false;

    /* --> End of setting params <-- */
    int ret = 0;

    // original pretrained model from https://github.com/AlexeyAB/darknet
    // the ncnn model https://drive.google.com/drive/folders/1YzILvh0SKQPS_lrb33dmGNq7aVTKPWS0?usp=sharing
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
#ifdef YOLOV4_TINY
    const char* classifier_param = param;
    const char* classifier_bin = bin;
    // *target_size = 608;
#else
    const char* classifier_param = "yolov4-opt.param";
    const char* classifier_model = "yolov4-opt.bin";
    // *target_size = 608;
#endif

    ret = classifier->load_param(classifier_param);
    if (ret != 0)
    {
        return ret;
    }

    ret = classifier->load_model(classifier_bin);
    if (ret != 0)
    {
        return ret;
    }

    return 0;
}

static int classify_cropped_image(const cv::Mat& cropped_image, int target_size, ncnn::Net* classifier)
{
    std::cout << "detect classification" << std::endl;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(cropped_image.data, ncnn::Mat::PIXEL_BGR2RGB, cropped_image.cols, cropped_image.rows, target_size, target_size);

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = classifier->create_extractor();
    printf("Here 1\n");

    ex.input("input_11_blob", in);
    printf("Here 2\n");

    ncnn::Mat out;
    ex.extract("dense_10_Softmax_blob", out);
    printf("after output\n");
    const float* values = NULL;
    for (int i = 0; i < out.h; i++)
    {
        values = out.row(i);
        for (int j = 0; j < sizeof(values) / sizeof(float); j++)
        {
            std::cout << "values[" << j <<  "]: " << values[j] << std::endl;
        }
    }
    if (values[0] > values[1])
    {
        printf("Deo\n");
        return 1;
    }
    else
    {
        printf("Khong deo\n");
        return 0;
    }
    // return (values[0] > values[1]);
}

std::string GetFileName(const std::string& s) {

    char sep = '/';

    size_t i = s.rfind(sep, s.length());
    std::string s1;
    if (i != std::string::npos) 
    {
        s1 = s.substr(i+1, s.length() - i);
    }

    sep = '.';
    i = s1.rfind(sep, s1.length());
    if (i != std::string::npos) 
    {
        return (s1.substr(0, i));
    }

    return("");
}

int pad_rectangle(cv::Mat image, cv::Rect_<float> &rect, int paddingPercent)
{
    float padding = (float)(paddingPercent) / 100;
    rect.x = rect.x - (rect.width * padding) / 2;
    rect.y = rect.y -  (rect.height * padding) / 2;
    rect.width = rect.width + (rect.width * padding);
    rect.height = rect.height + (rect.height * padding);

    if (rect.x < 0) rect.x = 0;
    if (rect.y < 0) rect.y = 0; 
    if ((rect.x + rect.width) > image.cols) rect.width = image.cols - rect.x;
    if ((rect.y + rect.height) > image.rows) rect.height = image.rows - rect.y;
}

int crop_all_face(cv::Mat &image, const std::vector<FaceObject> &face_objects, std::string name)
{
    for(size_t i=0; i<face_objects.size(); i++) {
        FaceObject obj = face_objects[i];
        pad_rectangle(image, obj.rect, 170);
        cv::Mat crop_iamge = image(obj.rect);
        cv::imwrite(name + "_object_" + std::to_string(i) + ".jpg", crop_iamge);
    }
    return 0;
}

int draw_hybrid_model(cv::Mat &rgb, const std::vector<FaceObject> &face_objects)
{
    // Object facemaskObject;
    // cv::namedWindow("show", cv::WINDOW_AUTOSIZE);
    for(size_t i=0; i<face_objects.size(); i++) {
        const FaceObject &obj = face_objects[i];
        cv::Rect_<float> rect = obj.rect;
        pad_rectangle(rgb, rect, 120);
        cv::Mat crop_image = rgb(rect);
        // cv::imshow("show", crop_image);
        // cv::waitKey(0);
        /*Detect mask by yolov4*/
        // yolov4_detect_cropped_image(crop_image.clone(), facemaskObject, target_size, &yolov4);

        // std::cout << "facemaskObject.label: " << facemaskObject.label << std::endl;
        // std::cout << "facemaskObject.porb: "  <<  facemaskObject.prob << std::endl;

        // if (facemaskObject.label == 1)
        // {
        //     cv::rectangle(rgb, rect, cv::Scalar(0, 0, 255), 2, 3);
        //     cv::putText(rgb, class_names[facemaskObject.label] + std::to_string(facemaskObject.prob), cv::Point(rect.x, rect.y + rect.height / 2),
        //     cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        // }
        // else if (facemaskObject.label == 2)
        // {
        //     // cv::rectangle(rgb, obj.rect, cv::Scalar(0, 255, 0), 2);
        //     cv::rectangle(rgb, rect, cv::Scalar(0, 255, 0), 2, 3);
        //     cv::putText(rgb, class_names[facemaskObject.label] + std::to_string(facemaskObject.prob), cv::Point(rect.x, rect.y + rect.height / 2),
        //     cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        // }
        // else break;

        /*Detect mask by classification model*/
        if (classify_cropped_image(crop_image.clone(), 32, &classifier))
        {
            // cv::rectangle(rgb, obj.rect, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(rgb, rect, cv::Scalar(0, 255, 0), 2, 3);
            // cv::putText(rgb, class_names[facemaskObject.label] + std::to_string(facemaskObject.prob), cv::Point(rect.x, rect.y + rect.height / 2),
            // cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        }
        else
        {
            cv::rectangle(rgb, rect, cv::Scalar(0, 0, 255), 2, 3);
            // cv::putText(rgb, class_names[facemaskObject.label] + std::to_string(facemaskObject.prob), cv::Point(rect.x, rect.y + rect.height / 2),
            // cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        }
    }
    return 0;
}


int main(int argc, char** argv)
{
    // scrfd.load(model_type, "/home/tienvh/Model/facemask_with_facedetection/faceDetectModel/");

    // std::cout << "Input all below parameters: " << std::endl;
    // std::cout << "Path to ncnn param: " << std::endl;
    // std::cin >> pathToYolov4Param;
    // std::cout << "Path to ncnn model(bin): " << std::endl;
    // std::cin >> pathToYolov4Model;
    // std::cout << "Target size: " << std::endl;
    // std::cin >> target_size;
    // std::cout << "Path to folfer image: " << std::endl;
    // std::cin >> pathToFolderImage;
    // std::cout << "Path to folder result: " << std::endl;
    // std::cin >> pathToFolderResult;

    int ret = init_yolov4(&yolov4, pathToParam.c_str(), pathToBin.c_str(), target_size); //We load model and param first!
    // int ret = init_classification_model(&classifier, pathToParam.c_str(), pathToBin.c_str());
    // if (ret != 0)
    // {
    //     fprintf(stderr, "Failed to load model or param, error %d", ret);
    //     return -1;
    // }
    std::string name;
    std::string name1;
    std::string name_new_file;
    std::string name_new_file1;
    Object object;
    std::vector<cv::String> fn;
    cv::glob(pathToFolderImage + "/*.jpg", fn, false);
    size_t count = fn.size();
    for(int i=0 ; i<= count ; i++ )
    {
        printf("Image %d:",i);
        printf("\n");
        name = fn[i];     
        name1= fn[i];
        name_new_file = GetFileName(fn[i]) + ".jpg";
        char cstr[name.size() + 1];
        strcpy(cstr,name.c_str());
        name_new_file1 = pathToFolderResult + "/" + name_new_file;
        char cstr_new[name_new_file1.size() + 1];

        frame = cv::imread(name, 1);

        //============================== scrfd model ================================
        // auto begin = std::chrono::high_resolution_clock::now();
        // scrfd.detect(frame, face_objects);
        // draw_hybrid_model(frame, face_objects);
        // crop_all_face(frame, face_objects, name_new_file1);
        // scrfd.draw(frame, face_objects);
        // if (face_objects.size())
        //     cv::imwrite(name_new_file1, frame);

        // auto end = std::chrono::high_resolution_clock::now();
        // auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        // printf("Time measured: %.3f seconds. ---------------------------------\n", elapsed.count() * 1e-9);

        //============================== model detect ===============================
        // retina.Detect(frame,face);
        // for (int i = 0; i < face.size(); i++)
        // {
        //     cv::rectangle(frame, cv::Point(face[i].x1, face[i].y1), 
        //                 cv::Point(face[i].x2, face[i].y2), cv::Scalar(0,255,0), 2);
        // }


        //============================= yolov4 facemask =============================
        if (frame.empty())
        {
            fprintf(stderr, "Failed to read image %s.\n", name_new_file.c_str());
            return -1;
        }
        detect_yolov4(frame, objects, target_size, &yolov4, name_new_file1); //Create an extractor and run detection
        // yolov4_detect_cropped_image(frame, object, target_size, &yolov4);
    }
    return 0;
}
