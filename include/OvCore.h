#ifndef AUTOAIM_OV_CORE_H
#define AUTOAIM_OV_CORE_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>

namespace aim {

void coutInfo(std::string, std::string);
void coutInfo(std::string, std::string, std::string);
} // namespace aim

namespace aim::openvino {

typedef void (*CoutInfo)(std::string);

/// @implements pack openvino's openvino deriviation
class AimCore {
  public:
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Core core;

    AimCore() = default;
    ~AimCore() = default;

    virtual void loadQuiet(std::string);
    virtual void loadVerbose(std::string);
    static void printModelInfo(std::shared_ptr<ov::Model>, CoutInfo);

  private:
    static void coutInfo(std::string);
};
} // namespace aim::openvino

#endif // AUTOAIM_OV_CORE_H