/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef SimpleFit_H
#define SimpleFit_H
#include "StatModule.h"
class SimpleFit : public StatModule {
 public:
  SimpleFit(const std::string &_name = "SimpleFit") : StatModule(_name) {}
  bool run(int = 0) final;
};
#endif
