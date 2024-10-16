#ifndef AUTOAIM_PROTO_H
#define AUTOAIM_PROTO_H

#include <opencv2/opencv.hpp>
#include <ostream>
#include <set>

namespace aim {

class ArmorRect : public cv::Rect2f {
  public:
    std::uint8_t bar_color;
    std::uint8_t pattern;
    float prob;

    ArmorRect() = default;
    ArmorRect(float, float, float, float, float, std::uint8_t, std::uint8_t);
    ~ArmorRect() = default;

    struct Cmp {
        bool operator()(const ArmorRect &, const ArmorRect &) const;
    };

    friend std::ostream &operator<<(std::ostream &os, const ArmorRect &am);
};

using ArmorRects = std::set<ArmorRect, ArmorRect::Cmp>;

} // namespace aim

#endif // AUTOAIM_PROTO_H