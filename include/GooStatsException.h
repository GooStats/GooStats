/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef GooStatsException_H
#define GooStatsException_H
#include <stdexcept>
#include "TSystem.h"
class GooStatsException : public std::runtime_error {
  public:
    GooStatsException(const std::string &what) : std::runtime_error(what) { gSystem->StackTrace(); }
};
#endif
