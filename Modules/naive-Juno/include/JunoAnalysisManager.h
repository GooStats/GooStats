/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef JunoAnalysisManager_H
#define JunoAnalysisManager_H
#include "AnalysisManager.h"
class JunoAnalysisManager : public AnalysisManager {
  public:
    JunoAnalysisManager() {}

    bool init() final;
    bool finish() final;
};
#endif
