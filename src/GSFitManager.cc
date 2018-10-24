/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "GSFitManager.h"
#include "InputManager.h"
#include <iostream>
#include <sys/time.h>
#include <sys/times.h>
#include "SumLikelihoodPdf.h"
bool GSFitManager::run(int) {
  sumpdf()->cache();
  restoreFitControl();
  fitManager = std::make_shared<FitManager>(sumpdf());
  fitManager->setMaxCalls(500000);
  clock_t startCPU, stopCPU; 
  timeval startTime, stopTime, totalTime;
  tms startProc, stopProc; 
  gettimeofday(&startTime, NULL);
  startCPU = times(&startProc);
  /*********** actual fit --> *************/
  fitManager->fit(); 
  fitManager->getMinuitValues();
  m_minim_conv = FitManager::minim_conv;
  m_hesse_conv = FitManager::hesse_conv;
  /*********** actual fit <-- *************/
  eval();
  stopCPU = times(&stopProc);
  gettimeofday(&stopTime, NULL);
  double myCPU = stopCPU - startCPU;
  double totalCPU = myCPU; 
  timersub(&stopTime, &startTime, &totalTime);
  std::cout << "Wallclock time  : " << totalTime.tv_sec + totalTime.tv_usec/1000000.0 << " seconds." << std::endl;
  std::cout << "CPU time: " << (myCPU / CLOCKS_PER_SEC) << std::endl; 
  std::cout << "Total CPU time: " << (totalCPU / CLOCKS_PER_SEC) << std::endl; 
  myCPU = stopProc.tms_utime - startProc.tms_utime;
  std::cout << "Processor time: " << (myCPU / CLOCKS_PER_SEC) << std::endl;
  sumpdf()->printProfileInfo(); 
  return true;
}
void GSFitManager::restoreFitControl() {
  if(LLfit())
    sumpdf()->setFitControl(new BinnedNllFit);
  else
    sumpdf()->setFitControl(new BinnedChisqFit);
}
InputManager *GSFitManager::getInputManager() {
  return static_cast<InputManager*>(find("InputManager"));
}
const InputManager *GSFitManager::getInputManager() const {
  return static_cast<const InputManager*>(find("InputManager"));
}
SumLikelihoodPdf *GSFitManager::sumpdf() {
  return getInputManager()->getTotalPdf();
}
bool GSFitManager::LLfit() const {
 return !getInputManager()->GlobalOption()->hasAndYes("chisquareFit");
}
#include "SumPdf.h"
void GSFitManager::eval() {
  // p-value
  toyMC.get_p_value(this,getInputManager(),minus2lnlikelihood()/2,m_LLp,m_LLpErr);
  // likelihood
  sumpdf()->setFitControl(new BinnedNllFit);
  sumpdf()->copyParams();
  m_likelihood = sumpdf()->calculateNLL();
  // chi2
  sumpdf()->setFitControl(new BinnedChisqFit);
  sumpdf()->copyParams();
  m_chi2 = sumpdf()->calculateNLL();
  // NDF
  m_NDF = 0;
  for(auto component : sumpdf()->Components()) {
    SumPdf *pdf = dynamic_cast<SumPdf*>(component);
    if(pdf) m_NDF += pdf->NDF();
  }
  restoreFitControl();
}
