#include "RT-DETRv2.h"
#include <cstdint>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/core/shape.hpp>

int main() {
    std::string model_path = "../../resources/model.onnx";
    aim::RTDETRv2 model{model_path, true};
    std::cout << "Successfully compile verbosely!\n";
    // aim::RTDETRv2 model{model_path};
    // std::cout << "Successfully compile quietly!\n";

    cv::Mat img = cv::imread("../../resources/cat.jpg");
    std::cout << img.type() << std::endl;
    // ov::Tensor tensor(ov::element::u8, ov::Shape({1, 640, 640, 3}), img.data);
    ov::Tensor tensor(ov::element::u8, ov::Shape({1, 640, 960, 3}));
    model.mat2ov(img, tensor);
    // for (auto i{1}; i <= tensor.get_size(); i ++) std::cout <<
    // *(tensor.data<std::uint8_t>() + i);
    cv::imshow("Image", img);
    cv::waitKey();

    std::vector<std::uint16_t> sz{640, 960};
    ov::Tensor size(ov::element::u16, ov::Shape({1, 2}), sz.data());
    // model.infer_request.set_input_tensors({tensor, size});
    model.infer_request.set_input_tensor(0, tensor);
    model.infer_request.set_input_tensor(1, size);
    model.infer_request.infer();

    auto output0 = model.infer_request.get_output_tensor(0);
    auto output1 = model.infer_request.get_output_tensor(1);
    auto output2 = model.infer_request.get_output_tensor(2);

    model.infer_request.cancel();
    return 0;
}