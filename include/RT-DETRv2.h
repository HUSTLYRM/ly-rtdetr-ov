#ifndef AUTOAIM_RT_DETR_v2_H
#define AUTOAIM_RT_DETR_v2_H

#include "OvCore.h"
#include "Proto.h"
#include <memory>
#include <ngraph/type/element_type.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>

namespace aim {

constexpr std::uint16_t rtdetrv2_input_height = 640;
constexpr std::uint16_t rtdetrv2_input_width = 640;


class RTDETRv2 : public openvino::AimCore {
public:
  // ov::Output<const ov::Node> input_port1, input_port2;

  RTDETRv2(std::string);
  RTDETRv2(std::string, bool);
  ~RTDETRv2() = default;

  void loadQuiet(std::string model_path) override {
    auto model = core.read_model(model_path);
    model = ppp(model);
    RTDETRv2::coutInfo("Using quiet compilation...");
    compiled_model = core.compile_model(model);
    infer_request = compiled_model.create_infer_request();
    // infer_request.infer();
    // RTDETRv2::getInputPort(false);
  }

  void loadVerbose(std::string model_path) override {
    auto model = core.read_model(model_path);
    model = ppp(model);
    RTDETRv2::coutInfo("Successfully preprocess!");
    AimCore::printModelInfo(model, coutInfo);
    compiled_model = core.compile_model(model);
    RTDETRv2::coutInfo("Successfully compile model!");
    infer_request = compiled_model.create_infer_request();
    // infer_request.infer();
    RTDETRv2::coutInfo("Successfully request inference!");
    // RTDETRv2::getInputPort(true);
    // RTDETRv2::coutInfo("Successfully get 2 input ports!");
  };

  void getInputPort(bool);
  static void coutInfo(std::string);
  void coutInfo(std::string, std::string);
  std::shared_ptr<ov::Model> ppp(std::shared_ptr<ov::Model>);
  void mat2ov(const cv::Mat &, ov::Tensor &);
  void infer(const cv::Mat &);
  void postproc(const ov::Tensor&, const ov::Tensor&, const ov::Tensor&, aim::ArmorRectDetSet&);
};

} // namespace aim

#endif // AUTOAIM_RT_DETR_v2_H