cmake_minimum_required(VERSION 3.2)
project(Benchii_ncnn CXX)

add_executable(Benchii_ncnn
   main.cpp)

# set(ncnn_DIR ../../cpp_libs/ncnn_wiiu/lib/cmake/ncnn CACHE PATH "Directory that contains ncnnConfig.cmake")
# find_package(ncnn REQUIRED)
# target_link_libraries(Benchii_ncnn ncnn)

wut_create_rpx(Benchii_ncnn)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/Benchii_ncnn.rpx"
        DESTINATION "${CMAKE_INSTALL_PREFIX}")
