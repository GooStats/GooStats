/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef DeclareModule_H
#define DeclareModule_H
class Module;
#include <string>
struct ModuleRegister {
  typedef Module* (*ModuleCreator)(const std::string&);
  ModuleRegister(const std::string& name, ModuleCreator creator);
};
#define DECLARE_MODULE(ModuleClass)                                                                    \
  Module* GooStats_##ModuleClass##_creator_(const std::string& name) { return new ModuleClass(name); } \
  ModuleRegister GooStats_register_Module_##ModuleClass##_(#ModuleClass, &GooStats_##ModuleClass##_creator_)
#endif
