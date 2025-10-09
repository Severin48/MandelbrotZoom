cmake_minimum_required(VERSION 3.16)
project(MellowSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Treat warnings seriously
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(MellowSim MellowSim.cpp)

target_link_libraries(MellowSim
        PRIVATE
        ${OpenCV_LIBS}
        OpenCL::OpenCL
)

set_target_properties(MellowSim PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(MellowSim PRIVATE -fsanitize=address,undefined)
        target_link_options(MellowSim PRIVATE -fsanitize=address,undefined)
    endif()
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fno-omit-frame-pointer -fno-sanitize-recover=all)
endif()

target_compile_definitions(MellowSim PRIVATE
        "PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\""
)