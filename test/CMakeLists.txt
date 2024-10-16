cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(AUTOAIM_TEST)

find_package(OpenCV 4 REQUIRED)
find_package(OpenVINO REQUIRED)

link_libraries(
    ${OpenCV_LIBS}
    openvino::runtime
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/../include
)

# add_executable(testOvCore
#     ${CMAKE_CURRENT_LIST_DIR}/testOvCore.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/../src/OvCore.cpp
# )

add_executable(testRTDETRv2
${CMAKE_CURRENT_LIST_DIR}/testRT-DETRv2.cpp
${CMAKE_CURRENT_LIST_DIR}/../src/OvCore.cpp
${CMAKE_CURRENT_LIST_DIR}/../src/RT-DETRv2.cpp
)

# add_executable(testProto
#     ${CMAKE_CURRENT_LIST_DIR}/testProto.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/../src/Proto.cpp
#     ${CMAKE_CURRENT_LIST_DIR}/../src/OvCore.cpp
# )