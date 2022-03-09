/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "Module.h"
const std::string Module::list() const {
  std::string names;
  for (auto dep : dependences) {
    names += dep.first + ",";
  }
  names = names.substr(0, names.size() - 1);
  return names;
}
