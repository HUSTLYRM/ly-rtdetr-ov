#ifndef AUTOAIM_CV_TRANS_H
#define AUTOAIM_CV_TRANS_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace aim::trans {

inline void hwc2ov(cv::Mat&);
inline void todtype(cv::Mat&);
inline void normalize(cv::Mat&);
}

#endif // AUTOAIM_CV_TRANS_H