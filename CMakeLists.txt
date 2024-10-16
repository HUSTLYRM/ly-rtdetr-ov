cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(AUTOAIM_RTDETR)

find_package(OpenCV 4 REQUIRED)
find_package(OpenVINO REQUIRED)

link_libraries(
    ${OpenCV_LIBS}
    openvino::runtime
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
)

add_executable(try
    ${CMAKE_CURRENT_LIST_DIR}/src/OvCore.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/RT-DETRv2.cpp
)
