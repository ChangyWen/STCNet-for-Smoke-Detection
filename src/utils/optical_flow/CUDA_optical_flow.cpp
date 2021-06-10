#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
using namespace std;

using namespace cv;
using namespace cv::cuda;

extern "C" void calc_cuda_flow (int rows, int cols, float *previous, float *current, float *flow){
  Ptr<cv::cuda::OpticalFlowDual_TVL1> opt = cv::cuda::OpticalFlowDual_TVL1::create();
	
  cv::Mat cpu_flow;
  cv::Mat previous_gray(rows, cols, CV_8UC1, (void*) previous);
  //cv::cuda::GpuMat gpu_prev_gray;
  //gpu_prev_gray.upload(previous_gray);

  cv::Mat current_gray(rows, cols, CV_8UC1, (void*) current);
  //cv::cuda::GpuMat gpu_curr_gray;
  //gpu_curr_gray.upload(current_gray);

  cv::Mat f(rows, cols, CV_8UC2, (void*) flow);
  //cv::cuda::GpuMat gpu_flow;
  //gpu_flow.upload(f);

  cv::cuda::GpuMat gpu_prev_gray;
  gpu_prev_gray.upload(previous_gray);
  cv::cuda::GpuMat gpu_curr_gray;
  gpu_curr_gray.upload(current_gray);
  cv::cuda::GpuMat gpu_flow;
  gpu_flow.upload(f);

  opt->calc(gpu_prev_gray, gpu_curr_gray, gpu_flow);
  //cv::Mat cpu_flow(gpu_flow);
  gpu_flow.download(cpu_flow);

  //uint8_t *data = cpu_flow.data;

  for(size_t i = 0; i < rows*cols*2; i++){
    float x = cpu_flow.at<float>(i);
    flow[i] = x;
  }
}
/*
  size_t index = 0;
  for(int i; i < rows; i++){
    for(int j; j < cols; j++){
      for(int k; k < 2; k++){
        float x = cpu_flow.at<float>(i,j,k);
        flow[index] = x;
        index += 1;
      }
    }
  }
}
*/

int main(int argc, char* argv[]){
  return 0;
}
