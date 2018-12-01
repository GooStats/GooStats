/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ModuleManager_H
#define ModuleManager_H
#include <map>
#include <string>
#include <memory>
class Module;
class ModuleManager {
  public:
    void registerModule(const std::string &type,const std::string &name="");
    bool initializeAllModules();
  private:
    bool initializeModule(Module *module);
    std::map<std::string,std::shared_ptr<Module>> modules;
};
#endif
