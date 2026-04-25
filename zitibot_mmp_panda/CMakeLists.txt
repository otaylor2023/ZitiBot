set(MMP_PANDA_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DMMP_PANDA_FOLDER="${MMP_PANDA_FOLDER}")

# create an executable (names distinct from project_starter/mmp_panda so both can be in one CMake tree)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CS225A_BINARY_DIR}/zitibot_mmp_example)
ADD_EXECUTABLE (controller_zitibot_mmp_panda controller.cpp ${CS225A_COMMON_SOURCE})
ADD_EXECUTABLE (simviz_zitibot_mmp_panda simviz.cpp ${CS225A_COMMON_SOURCE})

# and link the library against the executable
TARGET_LINK_LIBRARIES (controller_zitibot_mmp_panda ${CS225A_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (simviz_zitibot_mmp_panda ${CS225A_COMMON_LIBRARIES})
