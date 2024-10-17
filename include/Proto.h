#ifndef AUTOAIM_PROTO_H
#define AUTOAIM_PROTO_H

#include <opencv2/opencv.hpp>
#include <ostream>
#include <set>

namespace aim {

class ArmorRectDet : public cv::Rect2f {
public:
  std::uint8_t cls;
  float prob;

  ArmorRectDet() = default;
  ArmorRectDet(float, float, float, float, float, std::uint8_t);
  ~ArmorRectDet() = default;

  friend std::ostream &operator<<(std::ostream &os, const ArmorRectDet &am);
};

struct CmpArmorRectDet {
  bool operator()(const ArmorRectDet &, const ArmorRectDet &) const;
};

using ArmorRectDetSet = std::set<ArmorRectDet, CmpArmorRectDet>;

class ArmorRect : public cv::Rect2f {
public:
  std::uint8_t bar_color;
  std::uint8_t pattern;
  float prob;

  ArmorRect() = default;
  ArmorRect(float, float, float, float, float, std::uint8_t, std::uint8_t);
  ~ArmorRect() = default;

  friend std::ostream &operator<<(std::ostream &os, const ArmorRect &am);
};

struct CmpArmorRect {
  bool operator()(const ArmorRect &, const ArmorRect &) const;
};

using ArmorRectSet = std::set<ArmorRect, CmpArmorRect>;

} // namespace aim

#endif // AUTOAIM_PROTO_H