#include <opencv2/opencv.hpp>
// #include <opencv4/opencv2/opencv.hpp>
#include <iostream>  
#include <sstream> 
using namespace cv;
using namespace std;
int i=0;
int main(int argc, char** argv)
{
    int count = 1;
    int count_a=1;
    vector<cv::String> fn;
    string pathToFolderVideo = argv[1];
    string pathToFolder = argv[2];
    string imageName = argv[3];
    std::string pathToGlob = pathToFolderVideo + "/*.mp4";
    cout << "pathToGlob = " << pathToGlob << endl;
    glob(pathToGlob, fn, false);
    for (int i = 0; i < fn.size(); i++)
    {

        string pathToVideo = fn[i];
        
        // Open video file
            cout << argv[1] << endl;
            cout << "pathToVideo = " << pathToVideo << endl;
            VideoCapture cap(pathToVideo);

            int fps = cap.get(CAP_PROP_FPS);

            // For OpenCV 3, you can also use the following
            // double fps = video.get(CAP_PROP_FPS);

            cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;
            Mat frame; 

            while(true)
            {
                
                cap >> frame;
                if (frame.empty()) break;
                //imshow( "Frame", frame );
                string name = pathToFolder + "/" + imageName + "_" + to_string(count_a) + ".jpg";
                // std::string name = "/home/tienvh/BienVinhLong/VinhLong_"+std::to_string(count_a)+".jpg";
                
                if (count % fps ==0){
                    
                    cv::imwrite(name,frame);  
                    cout<<name<<endl;
                    count_a +=1;
                } 
                
                char c=(char)waitKey(40);
                count=count+1;
                if(c==32)
                break;
                
            }
        
        cap.release();
    }
    return 0;

}
