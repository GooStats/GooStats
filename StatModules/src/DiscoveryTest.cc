/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "DiscoveryTest.h"
#include "InputManager.h"
#include "GSFitManager.h"
#include "OutputManager.h"
bool DiscoveryTest::run(int ev) {
  if(!GlobalOption()->has("DiscoveryTest")) return true;
  auto parName = GlobalOption()->get("DiscoveryTest");
  auto var = getGSFitManager()->get_var(parName);
  var->fixed = true;
  var->value = 0;
  getGSFitManager()->run(ev);
  getOutputManager()->subFit(ev);
  var->fixed = false;
  var->value = (var->upperlimit+var->lowerlimit)/2;
  return true;
}
