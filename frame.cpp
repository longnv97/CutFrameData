#include <opencv2/opencv.hpp>
// #include <opencv4/opencv2/opencv.hpp>
#include <iostream>  
#include <sstream> 
using namespace cv;
using namespace std;
int i=0;
int main()
{
        for(int i = 1; i < 900; i++)
{
std::string path{"/home/longnve/AICam/CamAbility/LongNVe/Facetest/" + std::to_string(i) + ".jpg"};
cv::Mat img = cv::imread(path);
cv::resize(img, img, cv::Size(672,384));
std::string path_out{"/home/longnve/AICam/CamAbility/LongNVe/Facetest/1/" + std::to_string(i) + ".jpg"};
cv::imwrite(path_out, img); 
} 
    return 0;

}
