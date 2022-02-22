command build: 
g++ frame.cpp -o frame `pkg-config --libs opencv4`
￼
command run:
./frame arg1 arg2 arg3

where:
arg1: path to video
arg2: path to folder to store images
arg3: name for images

The video will be cut after each 2 seconds

