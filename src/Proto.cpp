#ifndef AUTOAIM_PROTO_CPP
#define AUTOAIM_PROTO_CPP

#include "Proto.h"
#include "OvCore.h"
#include <charconv>
#include <exception>
#include <ostream>
#include <string>

namespace aim {

ArmorRectDet::ArmorRectDet(float _x, float _y, float _w, float _h, float _prob,
                           std::uint8_t _cls) {
  if (_prob < 0.f || 1.f < _prob) {
    aim::coutInfo("ERROR", "ArmorRect", "Impossible prob!");
    std::terminate();
  }
  x = _x, y = _y, width = _w, height = _h, prob = _prob, cls = _cls;
}

std::ostream &operator<<(std::ostream &os, const ArmorRectDet &am) {
  os << "ArmorRectDet(" << am.tl() << "," << am.br()
     << "),prob=" << std::to_string(am.prob)
     << ",class=" << std::to_string(am.cls);
  return os;
}

bool CmpArmorRectDet::operator()(const ArmorRectDet &l,
                                 const ArmorRectDet &r) const {
  if (l.prob > r.prob) return true;
  if (l.prob < r.prob) return false;
  if (l.cls < r.cls) return true;
  if (l.cls > r.cls) return false;
  std::terminate();
}

ArmorRect::ArmorRect(float _x, float _y, float _w, float _h, float _prob,
                     std::uint8_t _bar, std::uint8_t _pat) {
  if (_prob < 0.f || 1.f < _prob) {
    aim::coutInfo("ERROR", "ArmorRect", "Impossible prob!");
    std::terminate();
  }
  x = _x, y = _y, width = _w, height = _h;
  prob = _prob, bar_color = _bar, pattern = _pat;
}

bool CmpArmorRect::operator()(const ArmorRect &l, const ArmorRect &r) const {
  if (l.prob > r.prob)
    return true;
  else if (l.prob < r.prob)
    return false;
  else if (l.pattern < r.pattern)
    return true;
  else if (l.pattern > r.pattern)
    return false;
  else if (l.bar_color < r.bar_color)
    return true;
  else if (l.bar_color > r.bar_color)
    return false;
  aim::coutInfo("ERROR", "ArmorRect", "Get 2 very similar armors!");
  std::terminate();
}

std::ostream &operator<<(std::ostream &os, const ArmorRect &am) {
  os << "ArmorRect(" << am.tl() << "," << am.br()
     << "),prob=" << std::to_string(am.prob)
     << ",pattern=" << std::to_string(am.pattern)
     << ",barcolor=" << std::to_string(am.bar_color);
  return os;
}
} // namespace aim

#endif // AUTOAIM_PROTO_CPP