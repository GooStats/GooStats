/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "SimpleFit.h"
#include "OptionManager.h"
#include "GSFitManager.h"
#include "OutputManager.h"
bool SimpleFit::run(int ev) {
  if(GlobalOption()->has("SimpleFit")&&!GlobalOption()->yes("SimpleFit")) return true;
  getGSFitManager()->run(ev);
  getOutputManager()->subFit(ev);
  return true;
}
