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
#include "Module.h"
bool AnalysisManager::init() {
  bool ok = true;
  for(auto mod : modules) {
    ok &= mod->preinit();
    if(!ok) throw GooStatsException("PreInit phase return false. check dependences of ["+mod->name()+"]");
  }
  for(auto mod : modules) 
    ok &= mod->init();
  if(!inputManager) throw GooStatsException("InputManager not ready. Please call AnalysisManager::setInputManager");
  return ok;
}
bool AnalysisManager::run(int event) {
  bool ok = true;
  for(auto mod : modules) 
    ok &= mod->run(event);
  return ok;
}
bool AnalysisManager::finish() {
  bool ok = true;
  for(auto mod : modules) 
    ok &= mod->finish();
  for(auto mod : modules) 
    ok &= mod->postfinish();
  return ok;
}
void AnalysisManager::setInputManager(InputManager *input) {
  inputManager = std::shared_ptr<InputManager>(input);
}
void AnalysisManager::setOutputManager(OutputManager *output) {
  outputManager = std::shared_ptr<OutputManager>(output);
}

bool AnalysisManager::registerModule(Module *module) {
  modules.push_back(std::shared_ptr<Module>(module));
  // remember to AnalysisManager::registerDependence before AnalysisManager::registerModule
  std::cout<<"["<<module->name()<<"]("<<module->list()<<") registered"<<std::endl;
  return true;
}
Module *AnalysisManager::findModule(const std::string &name) {
  for(auto mod : modules) 
    if(mod->name()==name) return mod.get();
  throw GooStatsException("Module <"+name+"> not found.");
}
bool AnalysisManager::hasModule(const std::string &name) const {
  for(auto mod : modules) 
    if(mod->name()==name) return true;
  return false;
}
