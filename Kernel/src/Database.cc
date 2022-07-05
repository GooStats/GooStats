/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Institute: Department of Physics, Princeton University
// Date: 2/28/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/

#include "Database.h"

#define DEFINE_METHOD(T, VAR)                                               \
  template <>                                                               \
  [[nodiscard]] const std::map<std::string, T> &Database::list<T>() const { \
    return VAR;                                                             \
  }                                                                         \
  template <>                                                               \
  [[nodiscard]] std::map<std::string, T> &Database::list<T>() {             \
    return VAR;                                                             \
  }

EXPAND_MACRO(DEFINE_METHOD);

template<>
void Database::set<std::string>(std::string key, std::string val, bool check) {
  std::string type = "string";
  if(key.find(":")!=std::string::npos) {
    auto head = GooStats::Utility::split(key, ":");
    if (head.size() != 2) {
      std::cout << "Cannot split <" << key << ">" << std::endl;
      throw GooStatsException("illegal format");
    }
    key = head.at(0);
    type = head.at(1);
  }
  if(type=="double") {
    setImpl<double>(key, std::stod(val), check);
  } else if(type=="int") {
    setImpl<int>(key,std::stoi(val),check);
  } else if(type=="string") {
    setImpl<std::string>(key,val,check);
  } else {
    std::cerr<<"Unkown type <"<<type<<">"<<std::endl;
    throw GooStatsException("illegal type");
  }
}
