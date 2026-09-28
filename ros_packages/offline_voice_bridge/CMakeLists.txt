cmake_minimum_required(VERSION 3.0.2)
project(offline_voice_bridge)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs)

catkin_package(CATKIN_DEPENDS rospy std_msgs)

catkin_install_python(PROGRAMS
  scripts/offline_tts_node.py
  scripts/offline_asr_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch tools config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
