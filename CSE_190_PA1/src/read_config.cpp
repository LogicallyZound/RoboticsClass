#include <fstream>
#include <iostream>
#include "../jsoncpp_example/json/json.h"
#include "../jsoncpp_example/jsoncpp.cpp"

Json::Value read_config() {
  Json::Value config;
  //Copy the configuration.json file in your working dir where you fire roslaunch & ADD the FULL path of the file below
  std::ifstream file("<Add PATH for configuration.json>");
  file >> config;
  return config;
}
