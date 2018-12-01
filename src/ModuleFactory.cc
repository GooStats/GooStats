/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ModuleFactory.h"
ModuleFactory *ModuleFactory::fModuleFactory = nullptr;
ModuleFactory *ModuleFactory::get() {
  if(!fModuleFactory) fModuleFactory = new ModuleFactory();
  return fModuleFactory;
}
Module *ModuleFactory::create(const std::string &type,const std::string &name) const {
  return modules.at(type)(name);
}
bool ModuleFactory::registerModule(const std::string &type,ModuleCreator creator) { 
  modules.insert(std::make_pair(type,creator));
  return true;
}
