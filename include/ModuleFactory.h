/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ModuleFactory_H
#define ModuleFactory_H
#include <map>
#include <memory>
#include <string>
class Module;
class ModuleFactory {
 public:
  static ModuleFactory *get();
  Module *create(const std::string &type, const std::string &name) const;

 private:
  typedef Module *(*ModuleCreator)(const std::string &);
  bool registerModule(const std::string &type, const ModuleCreator creator);
  friend struct ModuleRegister;

 private:
  static ModuleFactory *fModuleFactory;
  std::map<std::string, const ModuleCreator> modules;
};
#endif
