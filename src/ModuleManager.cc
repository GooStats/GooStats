/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ModuleManager.h"
#include "ModuleFactory.h"
#include "GooStatsException.h"
#include "Module.h"
void ModuleManager::registerModule(const std::string &type,const std::string &name) {
  const std::string &usedName(name==""?type:name);
  modules[usedName] = std::shared_ptr<Module>(ModuleFactory::get()->create(type,usedName));
}
bool ModuleManager::initializeAllModules() {
  for(auto module : modules) 
    if(!initializeModule(module.second.get())) 
      throw GooStatsException("Cannot initialize ("+module.first+")");
  return true;
}
bool ModuleManager::initializeModule(Module *module) {
  for(auto module_ : modules) {
    module->registerDependence(module_.second.get());
  }
  return true;
}
