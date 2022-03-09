/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ToyMCLikelihoodEvaluator_h
#define ToyMCLikelihoodEvaluator_h
#include <vector>

#include "goofit/FitControl.h"
class GSFitManager;
class InputManager;
class ToyMCLikelihoodEvaluator {
 public:
  void get_p_value(GSFitManager *gsFitManager,
                   InputManager *,
                   double LL,
                   double &p,
                   double &perr,
                   FitControl *fit = new BinnedNllFit());
  const std::vector<double> &getLLs() const { return LLs; }

 private:
  std::vector<double> LLs;
};
#endif
