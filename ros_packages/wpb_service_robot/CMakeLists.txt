cmake_minimum_required(VERSION 3.0.2)
project(wpb_service_robot)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  cv_bridge
  geometry_msgs
  move_base_msgs
  nav_msgs
  perception_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
)

catkin_package(
  CATKIN_DEPENDS actionlib actionlib_msgs cv_bridge geometry_msgs move_base_msgs nav_msgs perception_msgs rospy sensor_msgs std_msgs std_srvs
)

# 动作识别节点是通过 roslaunch 拉起来的（<node pkg="wpb_service_robot" .../>），
# 所以必须在这里登记，catkin 才会在 devel/lib/wpb_service_robot/ 下生成启动入口。
catkin_install_python(PROGRAMS
  scripts/qwen_action_recognition_node.py
  scripts/task_all_owners.py
  scripts/face_body_enroll_test.py
  scripts/owner_voice_reid_test.py
  scripts/person_reid_owner_test.py
  scripts/feedback_after_recognition.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch data models third_party
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
