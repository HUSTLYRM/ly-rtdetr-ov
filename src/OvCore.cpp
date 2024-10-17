#ifndef AUTOAIM_OV_CORE_CPP
#define AUTOAIM_OV_CORE_CPP

#include "OvCore.h"
#include <openvino/runtime/compiled_model.hpp>
#include <string>

namespace aim {

void coutInfo(std::string mod, std::string info) {
  std::cout << "[" << mod << "] " << info << "\n";
}

void coutInfo(std::string prefix, std::string mod, std::string info) {
  std::cout << "[" << prefix << "][" << mod << "] " << info << "\n";
}
} // namespace aim

namespace aim::openvino {

inline void AimCore::printModelInfo(std::shared_ptr<ov::Model> model,
                                    CoutInfo coutInfo) {
  coutInfo("Inference Model");
  coutInfo("  Model name: " + model->get_friendly_name());

  coutInfo("  Input:");
  for (auto input : model->inputs()) {
    coutInfo("    name: " + input.get_any_name());
    coutInfo("    type: " + input.get_element_type().c_type_string());
    coutInfo("    shape: " + input.get_partial_shape().to_string());
  }

  coutInfo("  Output:");
  for (auto output : model->outputs()) {
    coutInfo("    name: " + output.get_any_name());
    coutInfo("    type: " + output.get_element_type().c_type_string());
    coutInfo("    shape: " + output.get_partial_shape().to_string());
  }
}

void AimCore::loadQuiet(std::string model_path) {
  auto model = core.read_model(model_path);
  coutInfo("Using quiet compilation...");
  compiled_model = core.compile_model(model);
  infer_request = compiled_model.create_infer_request();
}

void AimCore::loadVerbose(std::string model_path) {
  auto model = core.read_model(model_path);
  printModelInfo(model, AimCore::coutInfo);
  compiled_model = core.compile_model(model);
  infer_request = compiled_model.create_infer_request();
}

void AimCore::coutInfo(std::string info) {
  // private
  aim::coutInfo("AimCore", info);
}
} // namespace aim::openvino

#endif // AUTOAIM_OV_CORE_CPP