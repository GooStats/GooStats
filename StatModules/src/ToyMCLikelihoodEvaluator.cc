/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ToyMCLikelihoodEvaluator.h"
#include "InputManager.h"
#include "SumLikelihoodPdf.h"
#include "GSFitManager.h"
void ToyMCLikelihoodEvaluator::get_p_value(GSFitManager *gsFitManager,InputManager *manager,double LL,double &p,double &perr,FitControl *fit) {
  const OptionManager *gOp = manager->GlobalOption();
  int N = gOp->has("toyMC_size")?gOp->get<double>("toyMC_size"):100;
  if(N==0) return;
  SumLikelihoodPdf *totalPdf = manager->getTotalPdf();
  totalPdf->setFitControl(fit);
  totalPdf->copyParams();
  totalPdf->cache();
  LLs.clear();
  int n = 0;
  for(int i = 0;i<N;++i) {
    manager->fillRandomData();
    LLs.push_back(totalPdf->calculateNLL());
    if(LLs.back()>LL) ++n;
  }
  totalPdf->restore();
  gsFitManager->restoreFitControl();
  p = n*1./N;
  perr = sqrt(p*(1-p)/N);
}
