#include "Proto.h"

int main() {
    aim::ArmorRect am1{};
    std::cout << am1 << "\n";
    aim::ArmorRect am2{0, 0, 1, 1, 0.5, 1, 16};
    std::cout << am2 << "\n";
    aim::ArmorRects ams;
    aim::ArmorRect am3{0, 0, 1, 1, 0.7, 1, 16};
    aim::ArmorRect am4{0, 0, 1, 1, 0.7, 2, 16};
    // aim::ArmorRect am5{0, 0, 1, 1, 0.2, 1, 16};
    // aim::ArmorRect am6{0, 0, 1, 1, 0.2, 8, 16};
    std::cout << "\n";
    ams.insert(am1);
    ams.insert(am2);
    ams.insert(am3);
    ams.insert(am4);
    for (auto it{ams.begin()}; it != ams.end(); it++) {
        std::cout << *it << "\n";
    }

    ams.insert({0, 0, 1, 1, 0.7, 2, 16});
    return 0;
}