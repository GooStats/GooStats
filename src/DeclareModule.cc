/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "DeclareModule.h"

#include "ModuleFactory.h"
ModuleRegister::ModuleRegister(const std::string &name, ModuleCreator creator) {
  ModuleFactory::get()->registerModule(name, creator);
}
