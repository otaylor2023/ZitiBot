# Standalone ZitiBot: cmake -S . -B build && cmake --build build
cmake_minimum_required(VERSION 3.10)
project(ZitiBot LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-std=c++17 -I/usr/include -I/usr/local/include -O3")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_SYSTEM_NAME MATCHES Darwin)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/include")
endif()

set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
if(NOT EIGEN3_INCLUDE_DIR)
	find_package(Eigen3 QUIET)
	if(NOT EIGEN3_INCLUDE_DIR)
		set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)
	endif()
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(CHAI3D REQUIRED)
include_directories(${CHAI3D_INCLUDE_DIRS})
add_definitions(${CHAI3D_DEFINITIONS})
if(CMAKE_SYSTEM_NAME MATCHES Darwin)
	add_definitions(-D_MACOSX -DMACOSX)
elseif(CMAKE_SYSTEM_NAME MATCHES Linux)
	add_definitions(-D_LINUX -DLINUX)
endif()
link_directories(${CHAI3D_LIBRARY_DIRS})

find_package(SAI-SIMULATION REQUIRED)
include_directories(${SAI-SIMULATION_INCLUDE_DIRS})
add_definitions(${SAI-SIMULATION_DEFINITIONS})

find_package(SAI-MODEL REQUIRED)
include_directories(${SAI-MODEL_INCLUDE_DIRS})
add_definitions(${SAI-MODEL_DEFINITIONS})

find_package(SAI-GRAPHICS REQUIRED)
include_directories(${SAI-GRAPHICS_INCLUDE_DIRS})
add_definitions(${SAI-GRAPHICS_DEFINITIONS})

find_package(SAI-COMMON REQUIRED)
include_directories(${SAI-COMMON_INCLUDE_DIRS})
add_definitions(${SAI-COMMON_DEFINITIONS})

find_package(SAI-URDF REQUIRED)
include_directories(${SAI-URDF_INCLUDE_DIRS})
add_definitions(${SAI-URDF_DEFINITIONS})

find_package(SAI-PRIMITIVES REQUIRED)
include_directories(${SAI-PRIMITIVES_INCLUDE_DIRS})
add_definitions(${SAI-PRIMITIVES_DEFINITIONS})

find_package(SAI-INTERFACES REQUIRED)
include_directories(${SAI-INTERFACES_INCLUDE_DIRS})

find_package(glfw3 QUIET)
find_library(GLFW_LIBRARY glfw)
find_library(HIREDIS_LIBRARY hiredis)

set(CS225A_COMMON_LIBRARIES
	${CHAI3D_LIBARIES}
	${SAI-INTERFACES_LIBRARIES}
	${HIREDIS_LIBRARY}
	${GLFW_LIBRARY}
	${SAI-GRAPHICS_LIBRARIES}
	${SAI-MODEL_LIBRARIES}
	${SAI-SIMULATION_LIBRARIES}
	${SAI-COMMON_LIBRARIES}
	${SAI-URDF_LIBRARIES}
	${CHAI3D_LIBRARIES}
	${HIREDIS_LIBRARY}
	${SAI-PRIMITIVES_LIBRARIES}
)

set(URDF_MODELS_FOLDER "${CMAKE_SOURCE_DIR}/urdf_models" CACHE PATH
	"Bundled robot/world URDFs under ZitiBot/urdf_models (mmp_panda, panda, test_objects). Set to an absolute path to override.")
add_definitions(-DCS225A_URDF_FOLDER="${URDF_MODELS_FOLDER}")

set(CS225A_BINARY_DIR "${CMAKE_SOURCE_DIR}/bin")
file(MAKE_DIRECTORY "${CS225A_BINARY_DIR}")

set(PROJECT_STARTER_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DPROJECT_STARTER_FOLDER="${PROJECT_STARTER_FOLDER}")

add_subdirectory(zitibot_mmp_panda)
