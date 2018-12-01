/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "NMOTest.h"
#include "InputManager.h"
#include "OutputManager.h"
#include "GSFitManager.h"
bool NMOTest::run(int) {
  if(!GlobalOption()->hasAndYes("fitNMO")) return true;
  auto deltaM2s = getInputManager()->Datasets().front()->get<std::vector<Variable*>>("deltaM2s");
  deltaM2s[1]->value = - deltaM2s[1]->value;
  deltaM2s[1]->lowerlimit = - deltaM2s[1]->upperlimit;
  deltaM2s[1]->upperlimit = - deltaM2s[1]->lowerlimit;
  getGSFitManager()->run(0);
  getOutputManager()->subFit(0);
  deltaM2s[1]->value = - deltaM2s[1]->value;
  deltaM2s[1]->lowerlimit = - deltaM2s[1]->upperlimit;
  deltaM2s[1]->upperlimit = - deltaM2s[1]->lowerlimit;
  getGSFitManager()->run(0);
  getOutputManager()->subFit(0);
  return true;
}
