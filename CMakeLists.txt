cmake_minimum_required(VERSION 3.24)
project(digits-nn-c C)

set(CMAKE_C_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/src)

option(OPTIMIZATIONS "Optimized Compilation" OFF)

if(OPTIMIZATIONS)
    add_compile_options(-O3 -march=native)
endif()

add_executable(digits-recognizer
        src/main.c
        src/nn_core.c
        src/activations.c
        src/loss.c
        src/data.c
        src/utils.h
)

target_link_libraries(digits-recognizer m)
