/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ScanPar_H
#define ScanPar_H
#include "StatModule.h"
class ScanPar : public StatModule {
 public:
  ScanPar(const std::string &_name = "ScanPar") : StatModule(_name) {}
  bool run(int = 0) final;
};
#endif
