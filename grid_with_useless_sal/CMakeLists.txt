cmake_minimum_required(VERSION 2.8)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

project( display )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}")
set(SOURCES Saliency.cpp ${PROJECT_SOURCE_DIR}/Saliency.h)
add_executable( display display.cpp ${SOURCES})
target_link_libraries( display ${OpenCV_LIBS} )
