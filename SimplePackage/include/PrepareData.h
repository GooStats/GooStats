/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef PrepareData_H
#define PrepareData_H
#include "StatModule.h"
class PrepareData : public StatModule {
  public:
    PrepareData(const std::string &_name="PrepareData") : StatModule(_name) { }
    bool init() final;
    bool run(int event = 0) final;
  private:
    int seed;
};
#endif
