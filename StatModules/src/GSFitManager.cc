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

#include <sys/time.h>
#include <sys/times.h>

#include <iostream>

#include "DeclareModule.h"
#include "InputManager.h"
#include "SumLikelihoodPdf.h"
DECLARE_MODULE(GSFitManager);
bool GSFitManager::run(int) {
  sumpdf()->cache();
  restoreFitControl();
  getFitManager()->setMaxCalls(20000);
  clock_t startCPU, stopCPU;
  timeval startTime, stopTime, totalTime;
  tms startProc, stopProc;
  gettimeofday(&startTime, NULL);
  startCPU = times(&startProc);
  m_likelihood = m_chi2 = m_LLp = m_LLpErr = -1e300;
  /*********** actual fit --> *************/
  getFitManager()->fit();
  /*********** actual fit <-- *************/
  eval();
  stopCPU = times(&stopProc);
  gettimeofday(&stopTime, NULL);
  double myCPU = stopCPU - startCPU;
  double totalCPU = myCPU;
  timersub(&stopTime, &startTime, &totalTime);
  std::cout << "Wallclock time  : " << totalTime.tv_sec + totalTime.tv_usec / 1000000.0 << " seconds." << std::endl;
  std::cout << "CPU time: " << (myCPU / CLOCKS_PER_SEC) << std::endl;
  std::cout << "Total CPU time: " << (totalCPU / CLOCKS_PER_SEC) << std::endl;
  myCPU = stopProc.tms_utime - startProc.tms_utime;
  std::cout << "Processor time: " << (myCPU / CLOCKS_PER_SEC) << std::endl;
  sumpdf()->printProfileInfo();
  return true;
}
void GSFitManager::restoreFitControl() {
  if (LLfit())
    sumpdf()->setFitControl(new BinnedNllFit{}, true);
  else
    sumpdf()->setFitControl(new BinnedChisqFit{}, true);
}
SumLikelihoodPdf *GSFitManager::sumpdf() { return getInputManager()->getTotalPdf(); }
const SumLikelihoodPdf *GSFitManager::sumpdf() const { return getInputManager()->getTotalPdf(); }
bool GSFitManager::LLfit() const { return !GlobalOption()->hasAndYes("chisquareFit"); }
int GSFitManager::get_id(const std::string &parName) const {
  unsigned int counter = 0;
  std::vector<Variable *> vars;
  sumpdf()->getParameters(vars);
  for (auto var : vars) {
    if (var->name == parName)
      break;
    counter++;
  }
  if (counter == vars.size()) {
    counter = 0;
    for (std::vector<Variable *>::iterator i = vars.begin(); i != vars.end(); ++i) {
      std::cout << counter << " [" << ((*i)->name) << "]" << std::endl;
      counter++;
    }
    std::cout << "[" << parName << "] not found in (" << vars.size() << ") pars." << std::endl;
    throw GooStatsException("par not found");
  }
  return counter + 1;
}
Variable *GSFitManager::get_var(const std::string &parName) {
  std::vector<Variable *> vars;
  sumpdf()->getParameters(vars);
  for (auto var : vars) {
    if (var->name == parName)
      return var;
  }
  for (auto var : vars) {
    std::cout << "[" << var->name << "]" << std::endl;
  }
  std::cout << "[" << parName << "] not found in (" << vars.size() << ") pars." << std::endl;
  throw GooStatsException("par not found");
  return nullptr;
}
#include "goofit/PDFs/SumPdf.h"
void GSFitManager::eval() {
  getFitManager()->getMinuitValues();
  m_minim_conv = FitManager::minim_conv;
  m_hesse_conv = FitManager::hesse_conv;
  // likelihood
  sumpdf()->setFitControl(new BinnedNllFit{}, true);
  sumpdf()->copyParams();
  m_likelihood = sumpdf()->calculateNLL();
  // p-value
  toyMC.get_p_value(this, getInputManager(), minus2lnlikelihood() / 2, m_LLp, m_LLpErr);
  // chi2
  sumpdf()->setFitControl(new BinnedChisqFit{}, true);
  sumpdf()->copyParams();
  m_chi2 = sumpdf()->calculateNLL();
  // NDF
  int NnonzeroBins = 0;
  for (auto component : sumpdf()->Components()) {
    DataPdf *pdf = dynamic_cast<DataPdf *>(component);
    if (pdf)
      NnonzeroBins += pdf->NDF() + pdf->Nfree();
  }
  m_NDF = NnonzeroBins - sumpdf()->Nfree();
  restoreFitControl();
}
FitManager *GSFitManager::getFitManager() {
  if (fitManager)
    return fitManager.get();
  if (!sumpdf())
    throw GooStatsException("SumLikelihoodPdf not ready for GSFitManager::getMinuitValues");
  fitManager = std::make_shared<FitManager>(sumpdf());
  fitManager->setupMinuit();
  return fitManager.get();
}
