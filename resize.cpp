#include <opencv2/opencv.hpp>
// #include <opencv4/opencv2/opencv.hpp>
#include <iostream>  
#include <sstream> 
using namespace cv;
using namespace std;
int i=0;
int main(int argc, char** argv)
{
    string path_input = argv[1];
    string path_output = argv[2];
    
    for (int i = 1; i < 102; i++)
	{
		std::string path{path_input + "/" + std::to_string(i) + ".jpg"};
		cv::Mat img = cv::imread(path);
		cv::resize(img, img, cv::Size(672,384));
		std::string path_out{path_output + "/" + std::to_string(i) + ".jpg"};
		cv::imwrite(path_out, img); 	
	}
    
    return 0;

}
