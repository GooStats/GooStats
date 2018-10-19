/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ReactorAnalysisManager_H
#define ReactorAnalysisManager_H
#include "AnalysisManager.h"
class ReactorAnalysisManager : public AnalysisManager {
  public:
    ReactorAnalysisManager() {}

    bool run(int event = 0) final;
    bool finish() final;
};
#endif
