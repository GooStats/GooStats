/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "AnalysisManager.h"
#include "GooStatsException.h"
#include "InputManager.h"
#include "InputBuilder.h"
#include "ParSyncManager.h"
#include "OutputManager.h"
Bit AnalysisManager::getBit() { 
  static Bit bit = 0; 
  if(bit) bit <<= 1; else bit = 1; return bit; 
}
bool AnalysisManager::init() {
  checkGPU(); // check if we have GPU..
  inputManager->init();
  inputManager->initialize_configsets();
  inputManager->fill_rawSpectrumProvider();
  inputManager->initialize_controllers();
  inputManager->initialize_datasets();
  inputManager->buildTotalPdf();
  if(outputManager) outputManager->adoptInputManager(inputManager.get());
  if(outputManager) outputManager->init();

  sumpdf = inputManager->getTotalPdf();
  return true;
}
#include "goofit/FitManager.h"
#include <iostream>
#include <sys/time.h>
#include <sys/times.h>
bool AnalysisManager::run() {
  fitManager = std::make_shared<FitManager>(sumpdf);
  fitManager->setMaxCalls(500000);
  clock_t startCPU, stopCPU; 
  timeval startTime, stopTime, totalTime;
  tms startProc, stopProc; 
  gettimeofday(&startTime, NULL);
  startCPU = times(&startProc);
  fitManager->fit(); 
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
  sumpdf->printProfileInfo(); 
  fitManager->getMinuitValues();

  if(outputManager) outputManager->run();
  return true;
}
bool AnalysisManager::finish() {
  if(outputManager) outputManager->finish();
  return true;
}
void AnalysisManager::setInputManager(InputManager *input) {
  inputManager = std::shared_ptr<InputManager>(input);
}
void AnalysisManager::setOutputManager(OutputManager *output) {
  outputManager = std::shared_ptr<OutputManager>(output);
}

bool AnalysisManager::checkGPU() const {
  if(!GPUManager::get()->report()) {
    throw GooStatsException("Cannot find a free GPU!");
  } else
    return true;
}
#include "Module.h"
bool AnalysisManager::registerModule(Module *module) {
  for(auto mod : m_modules) 
    if(mod->dependence() & module->category()) 
      throw GooStatsException("<"+module->name()+"> is required by <"+mod->name()+"> and should be inserted before it.");
  m_modules.push_back(std::shared_ptr<Module>(module));
  return true;
}
