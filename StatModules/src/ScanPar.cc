/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ScanPar.h"

#include "GSFitManager.h"
#include "InputManager.h"
#include "OutputManager.h"
bool ScanPar::run(int) {
  if (!GlobalOption()->has("scanPar"))
    return true;
  auto parName = GlobalOption()->get("scanPar");
  auto var = getGSFitManager()->get_var(parName);
  auto left = GlobalOption()->get<double>("scanParMin");
  auto right = GlobalOption()->get<double>("scanParMax");
  int Npoint = GlobalOption()->get<double>("scanParN");
  var->fixed = true;
  for (int i = 0; i < Npoint; ++i) {
    double xx = left + (right - left) / (Npoint - 1) * i;
    var->value = xx;
    getGSFitManager()->run(i);
    getOutputManager()->subFit(i);
    printf("Scanning [%10s] (%3d/%3d;%5.2lf,%5.2lf,%5.2lf) -> %10.2lf\n",
           parName.c_str(),
           i,
           Npoint,
           xx,
           left,
           right,
           getGSFitManager()->minus2lnlikelihood());
  }
  var->fixed = false;
  var->value = (var->upperlimit + var->lowerlimit) / 2;
  return true;
}
