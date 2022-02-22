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
        for(int i = 1; i < 900; i++)
	{
		std::string path{path_input + "/" + std::to_string(i) + ".jpg"};

		std::string comment{"ffmpeg -y -i " + path + " -pix_fmt nv12 " + path_output + "/" + std::to_string(i) + ".yuv"};
		system(comment.c_str());
	} 
    return 0;

}
