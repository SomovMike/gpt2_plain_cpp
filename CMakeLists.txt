cmake_minimum_required(VERSION 3.15)
project(gpt2_plain_cpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(WITH_OPENMP "Compile with multithreading support" OFF)


if (WITH_OPENMP)
    if (APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
        set(OpenMP_C_FLAGS “-Xpreprocessor -fopenmp”)
        set(OpenMP_C_LIB_NAMES "libomp")
        set(OpenMP_CXX_FLAGS “-Xpreprocessor -fopenmp”)
        set(OpenMP_CXX_LIB_NAMES "libomp")
        set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_CXX_FOUND TRUE)
        set(OpenMP_C_FOUND TRUE)
        set(OPENMP_LIBRARIES "/opt/homebrew/opt/libomp/lib")
        set(OPENMP_INCLUDES "/opt/homebrew/opt/libomp/include")
        include_directories("${OPENMP_INCLUDES}")
        link_directories("${OPENMP_LIBRARIES}")
    endif()
    find_package(OpenMP)
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -O3 -ffast-math -Ofast -march=native")
    set(CMAKE_UNITY_BUILD ON)
    message("Release mode enabled")
endif()

add_executable(gpt2_plain_cpp
        run.cpp
        utils.cpp
        utils.h
        run.h)

if(WITH_OPENMP)
    target_compile_definitions(gpt2_plain_cpp PUBLIC WITH_OPENMP)
    if (APPLE)
        target_link_libraries(gpt2_plain_cpp PUBLIC omp)
    else ()
        target_link_libraries(gpt2_plain_cpp PUBLIC OpenMP::OpenMP_CXX)
    endif ()
endif ()

