#include "RT-DETRv2.h"
#include <cstdint>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <openvino/core/shape.hpp>
#include <opencv2/opencv.hpp>

int main() {
    std::string model_path = "../../resources/rtdetrv2_r18.onnx";
    aim::RTDETRv2 model{model_path, true};
    std::cout << "Successfully compile verbosely!\n";
    // aim::RTDETRv2 model{model_path};
    // std::cout << "Successfully compile quietly!\n";

    cv::Mat img = cv::imread("../../resources/kite.jpg");
    
    model.infer(img);


    // model.infer_request.cancel();
    return 0;
}