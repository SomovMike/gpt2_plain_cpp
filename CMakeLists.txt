cmake_minimum_required(VERSION 3.15)
project(gpt2_plain_cpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -O3 -ffast-math -Ofast -march=native")
    set(CMAKE_UNITY_BUILD ON)
    message("Release mode enabled")
endif()

add_executable(gpt2_plain_cpp
        run.cpp
        utils.cpp
        utils.h)
