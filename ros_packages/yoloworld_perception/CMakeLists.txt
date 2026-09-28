cmake_minimum_required(VERSION 3.0.2)
project(yoloworld_perception)

find_package(catkin REQUIRED)

catkin_package()

# 只提供模型文件，交给其他包在运行时通过 $(find yoloworld_perception) 取用
install(DIRECTORY models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
