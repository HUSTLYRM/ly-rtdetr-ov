#include "RT-DETRv2.h"
#include "OvCore.h"
#include "Proto.h"
#include <cstdint>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <openvino/core/dimension.hpp>
#include <openvino/core/preprocess/color_format.hpp>
#include <openvino/core/preprocess/input_info.hpp>
#include <openvino/core/preprocess/output_info.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/core/preprocess/resize_algorithm.hpp>
#include <string>

namespace aim {

RTDETRv2::RTDETRv2(std::string model_path) { RTDETRv2(model_path, false); }
RTDETRv2::RTDETRv2(std::string model_path, bool is_verbose) {
  if (is_verbose)
    loadVerbose(model_path);
  else
    loadQuiet(model_path);
}

void RTDETRv2::coutInfo(std::string info) { aim::coutInfo("RTDETRv2", info); }

void RTDETRv2::coutInfo(std::string prefix, std::string info) {
  aim::coutInfo(prefix, "RTDETRv2", info);
}

std::shared_ptr<ov::Model> RTDETRv2::ppp(std::shared_ptr<ov::Model> model) {
  ov::preprocess::PrePostProcessor ppp(model);
  ov::preprocess::InputInfo &images = ppp.input(0);
  ov::preprocess::InputInfo &orig_target_sizes = ppp.input(1);
  ov::preprocess::OutputInfo &labels = ppp.output(0);

  // input tensor match cv::Mat
  images.tensor().set_element_type(ov::element::u8);
  RTDETRv2::coutInfo("Input 0 -> u8");
  images.tensor().set_layout("NHWC");
  RTDETRv2::coutInfo("Input 0 -> NHWC");
  images.tensor().set_shape({1, -1, -1, 3});
  RTDETRv2::coutInfo("Input 0 -> dynamic shape");
  images.tensor().set_color_format(ov::preprocess::ColorFormat::BGR);
  RTDETRv2::coutInfo("Input 0 -> BGR");

  images.model().set_layout("NCHW");
  images.preprocess().convert_color(ov::preprocess::ColorFormat::RGB);
  // images.preprocess().scale(255);
  images.preprocess().resize(
      ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);

  orig_target_sizes.tensor().set_element_type(ov::element::u16);
  RTDETRv2::coutInfo("Input 1 -> u16");
  orig_target_sizes.tensor().set_shape({1, 2});
  RTDETRv2::coutInfo("Input 1 -> [1,2]");

  labels.tensor().set_element_type(ov::element::u8);
  RTDETRv2::coutInfo("Output 0 -> u8");

  return ppp.build();
}

void RTDETRv2::infer(const cv::Mat &mat) {
  infer_request.reset_state();
  auto img = infer_request.get_input_tensor(0);
  coutInfo("Get img w = " + std::to_string(mat.cols));
  coutInfo("Get img h = " + std::to_string(mat.rows));
  coutInfo("Get img type = " + std::to_string(mat.type()));
  coutInfo("Input img w = " + std::to_string(img.get_shape()[1]));
  coutInfo("Input img h = " + std::to_string(img.get_shape()[2]));
  coutInfo("Input img type = " + img.get_element_type().to_string());

  std::uint8_t *input_img_data = (std::uint8_t *)mat.data;
  ov::Tensor input_img =
      ov::Tensor(infer_request.get_input_tensor(0).get_element_type(),
                 ov::Shape{1, static_cast<unsigned long>(mat.rows),
                           static_cast<unsigned long>(mat.cols), 3},
                 input_img_data);
  infer_request.set_input_tensor(0, input_img);

  std::uint16_t input_size_data[2] = {static_cast<uint16_t>(mat.rows), static_cast<uint16_t>(mat.cols)};
  ov::Tensor input_size =
      ov::Tensor(infer_request.get_input_tensor(1).get_element_type(),
                 ov::Shape{1, 2}, input_size_data);
  infer_request.set_input_tensor(1, input_size);

  infer_request.infer();
  auto output0 = infer_request.get_output_tensor(0);
  auto output1 = infer_request.get_output_tensor(1);
  auto output2 = infer_request.get_output_tensor(2);

  aim::ArmorRectDetSet set;
  postproc(output0, output1, output2, set);
}

void RTDETRv2::postproc(const ov::Tensor &labels, const ov::Tensor &bboxes,
                        const ov::Tensor &scores,
                        aim::ArmorRectDetSet &armors) {
  auto labels_data = labels.data<std::uint8_t>();
  auto bboxes_data = bboxes.data<float>();
  auto scores_data = scores.data<float>();

  for (size_t i{0}; i < 300; i++) {
    armors.insert(aim::ArmorRectDet{*bboxes_data, *(bboxes_data + 1),
                                    *(bboxes_data + 2) - *bboxes_data, *(bboxes_data + 3) - *(bboxes_data + 1),
                                    *scores_data, *labels_data});
    labels_data++;
    bboxes_data += 4;
    scores_data++;
  }
  for (auto am : armors) {
    std::cout << am << "\n";
  }
}
} // namespace aim

// #ifndef AUTOAIM_RT_DETR_V2_CPP
// #define AUTOAIM_RT_DETR_V2_CPP

// #include "RT-DETRv2.h"
// #include "OvCore.h"
// #include <cstdint>
// #include <opencv2/core.hpp>
// #include <opencv2/core/types.hpp>
// #include <string>

// namespace aim {

// void RTDETRv2::detectArmor(cv::Mat &src, ) { return; }

// bool ArmorDetector::detect(cv::Mat &src, std::vector<ArmorObject> &objects,
//                            bool use_roi) {
//   if (src.empty()) {
//     return false;
//   }

//   cv::Mat pr_img = scaledResize(src, transfrom_matrix);
//   // dw = this->dw;

//   cv::Mat pre;
//   cv::Mat pre_split[3];
//   pr_img.convertTo(pre, CV_32F);
//   cv::split(pre, pre_split);

//   // Get input tensor by index
//   input_tensor = infer_request.get_input_tensor(0);

//   // 准备输入
//   infer_request.set_input_tensor(input_tensor);

//   float *tensor_data = input_tensor.data<float_t>();
//   // u_int8_t* tensor_data = input_tensor.data<u_int8_t>();

//   auto img_offset = INPUT_H * INPUT_W;
//   // Copy img into tensor
//   for (int c = 0; c < 3; c++) {
//     memcpy(tensor_data, pre_split[c].data, INPUT_H * INPUT_W *
//     sizeof(float));
//     // memcpy(tensor_data, pre_split[c].data, INPUT_H * INPUT_W *
//     // sizeof(u_int8_t));
//     tensor_data += img_offset;
//   }

//   // ov::element::Type input_type = ov::element::f32;
//   // ov::Shape input_shape = {1, 3, 416, 416};

//   // // std::shared_ptr<unsigned char> input_data_ptr = pre.data;
//   // auto input_data_ptr = pre.data;

//   // // 转换图像数据为ov::Tensor
//   // input_tensor = ov::Tensor(input_type, input_shape, input_data_ptr);

//   // auto st = std::chrono::steady_clock::now();
//   // 推理
//   infer_request.infer();
//   // auto end = std::chrono::steady_clock::now();
//   // double infer_dt = std::chrono::duration<double,std::milli>(end -
//   // st).count(); cout << "infer_time:" << infer_dt << endl;

//   // 处理推理结果
//   ov::Tensor output_tensor = infer_request.get_output_tensor();
//   float *output = output_tensor.data<float_t>();
//   // cout << "output:" << " ";
//   // for (int ii = 0; ii < 25; ii++)
//   // {
//   //     cout << output[ii] << " ";
//   // }
//   // cout << endl;
//   // u_int8_t* output = output_tensor.data<u_int8_t>();
//   // std::cout << &output << std::endl;

//   // int img_w = src.cols;
//   // int img_h = src.rows;
//   decodeOutputs(output, objects, transfrom_matrix);
//   for (auto object = objects.begin(); object != objects.end(); ++object) {
//     // 对候选框预测角点进行平均,降低误差
//     if ((*object).pts.size() >= 8) {
//       auto N = (*object).pts.size();
//       cv::Point2f pts_final[4];
//       for (int i = 0; i < (int)N; i++) {
//         pts_final[i % 4] += (*object).pts[i];
//       }

//       for (int i = 0; i < 4; i++) {
//         pts_final[i].x = pts_final[i].x / (N / 4);
//         pts_final[i].y = pts_final[i].y / (N / 4);
//       }

//       if (use_roi) { // 使用ROI后，需要将坐标移动到对应的位置,
//                      // 还原到对应的位置上
//         // std::cout << "use_roi" <<std::endl;
//         (*object).apex[0] =
//             pts_final[0] + cv::Point2f(roi_outpost_xmin, roi_outpost_ymin);
//         (*object).apex[1] =
//             pts_final[1] + cv::Point2f(roi_outpost_xmin, roi_outpost_ymin);
//         (*object).apex[2] =
//             pts_final[2] + cv::Point2f(roi_outpost_xmin, roi_outpost_ymin);
//         (*object).apex[3] =
//             pts_final[3] + cv::Point2f(roi_outpost_xmin, roi_outpost_ymin);
//       } else { // 防抖
//         (*object).apex[0] = pts_final[0];
//         (*object).apex[1] = pts_final[1];
//         (*object).apex[2] = pts_final[2];
//         (*object).apex[3] = pts_final[3];
//       }

//     } else {
//       if (use_roi) { // 使用ROI后，需要将坐标移动到对应的位置,
//                      // 还原到对应的位置上
//         // std::cout << "use_roi" <<std::endl;
//         (*object).apex[0] =
//             (*object).apex[0] + cv::Point2f(roi_outpost_xmin,
//             roi_outpost_ymin);
//         (*object).apex[1] =
//             (*object).apex[1] + cv::Point2f(roi_outpost_xmin,
//             roi_outpost_ymin);
//         (*object).apex[2] =
//             (*object).apex[2] + cv::Point2f(roi_outpost_xmin,
//             roi_outpost_ymin);
//         (*object).apex[3] =
//             (*object).apex[3] + cv::Point2f(roi_outpost_xmin,
//             roi_outpost_ymin);
//       }
//     }

//     // cv::Point2f pts_final[4];
//     // for(int ii = 0; ii < 4; ii++)
//     // {
//     //     pts_final[ii] = (*object).pts[ii];
//     // }
//     // (*object).apex[0] = pts_final[0];
//     // (*object).apex[1] = pts_final[1];
//     // (*object).apex[2] = pts_final[2];
//     // (*object).apex[3] = pts_final[3];

//     // cout << "output:";
//     // for (int i = 0; i < 4; i++)
//     // {
//     //     cout << " " << "(" << (*object).apex[i].x << "," <<
//     //     (*object).apex[i].y << ") ";
//     // }
//     // cout << "obj_prob:" << (*object).prob << " obj_color:" <<
//     (*object).color
//     // << " obj_cls:" << (*object).cls << endl;
//     (*object).area = (int)(calcTetragonArea((*object).apex));
//   }

//   if (objects.size() != 0)
//     return true;
//   else
//     return false;
// }

// static std::string colorText[] = {"Blue", "Red", "Gray", "Purple"};
// static std::string typeText[] = {"Sentry", "1", "2",       "3",
//                                  "4",      "5", "OutPost", "Base"};

// // 0 3
// // 1 2

// void ArmorDetector::drawArmors(cv::Mat &drawing,
//                                std::vector<ArmorObject> &objects) {
//   for (ArmorObject object : objects) {
//     // for(int i=0;i<4;i++){
//     //     cv::line(drawing, object.apex[i%4],
//     //     object.apex[(i+1)%4],cv::Scalar(0,0,255), 2);
//     // }
//     cv::line(drawing, object.apex[0], object.apex[1], cv::Scalar(0, 255, 0),
//              2); // 红
//     cv::line(drawing, object.apex[1], object.apex[3], cv::Scalar(0, 255, 0),
//              2); // 绿
//     cv::line(drawing, object.apex[3], object.apex[2], cv::Scalar(0, 255, 0),
//              2); // 蓝
//     cv::line(drawing, object.apex[2], object.apex[0], cv::Scalar(0, 255, 0),
//              2); // 白

//     // (blue / red / none / purple)  (small / big)
//     cv::putText(drawing,
//                 colorText[(object.color / 2)] + " " + typeText[object.cls] +
//                     " : " + std::to_string(object.prob * 100),
//                 object.apex[0], cv::FONT_HERSHEY_SIMPLEX, 1.5,
//                 cv::Scalar(0, 255, 0));
//   }
// }

// void RTDETRv2::coutInfo(std::string info) {
//   return aim::coutInfo("RTDETRv2", info);
// }

// void RTDETRv2::getInputport(void) {
//   input_port1 = compiled_model.input(0);
//   RTDETRv2::coutInfo(input_port1.get_any_name());
//   input_port2 = compiled_model.input(1);
//   RTDETRv2::coutInfo(input_port2.get_any_name());
// }

// /// @brief setStatic is fucked
// // void RTDETRv2::setStatic(std::shared_ptr<ov::Model> model) {
// //   std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;

// //   const ov::Output<ov::Node> &input0 = model->input(0);
// //   port_to_shape[input0] = {
// //       aim::rtdetrv2_input_batch_size, aim::rtdetrv2_input_channel,
// //       aim::rtdetrv2_input_height, aim::rtdetrv2_input_width};
// //   coutInfo(input0.get_partial_shape().to_string());

// //   const ov::Output<ov::Node> &input1 = model->input(1);
// //   port_to_shape[input1] = {aim::rtdetrv2_original_image_height,
// //                            aim::rtdetrv2_original_image_width};
// //   coutInfo(input1.get_partial_shape().to_string());

// //   model->reshape(port_to_shape);
// //   coutInfo("Successfully reshape!");
// // };

// } // namespace aim

// #endif // AUTOAIM_RT_DETR_V2_CPP