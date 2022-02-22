
# Copyright (C) 2020, LE MANH THANG. All rights reserved.
# Module: CMakelists.txt
# Author: LE MANH THANG
# Created: 10/01/2021
# Description: 
cmake_minimum_required(VERSION 3.0)

enable_language(C)
enable_language(CXX)

project (tracking)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


#add lib ncnn
set( NCNN_LIB libncnn.a)
link_directories(${CMAKE_SOURCE_DIR}/libncnn-linux_x86_64/lib)
include_directories( ${CMAKE_SOURCE_DIR}/libncnn-linux_x86_64/include/ncnn)

#add lib opencv
set( OPENCV_LIB libopencv_world.so)
link_directories(${CMAKE_SOURCE_DIR}/libopencv-4.5.0-linux-x64-prebuilt/lib)
include_directories( ${CMAKE_SOURCE_DIR}/libopencv-4.5.0-linux-x64-prebuilt/include/opencv4)

#add lib opencv
set( OPENCV_HASH_LIB libopencv_img_hash.so)
link_directories(${CMAKE_SOURCE_DIR}/libopencv-4.5.0-linux-x64-prebuilt/lib)
include_directories( ${CMAKE_SOURCE_DIR}/libopencv-4.5.0-linux-x64-prebuilt/include/opencv4)

file( GLOB SOURCES 	facemask.cpp detect.cpp scrfd.cpp)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/ncnn)

#add_library(bkav_Ai SHARED ${SOURCES})
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
${NCNN_LIB}
${OPENCV_LIB}
${OPENCV_HASH_LIB}
-Wall -lm -lpthread -lstdc++ -std=c++11 -lgomp
) 
