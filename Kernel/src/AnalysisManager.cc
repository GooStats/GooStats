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

#include <iostream>

#include "GPUManager.h"
#include "GooStatsException.h"
#include "Module.h"
AnalysisManager::AnalysisManager() { registerModule(std::unique_ptr<GPUManager>(new GPUManager{})); }
bool AnalysisManager::init() {
  bool ok = true;
  for (auto &mod : modules) {
    ok &= mod->preinit();
    if (!ok)
      throw GooStatsException("PreInit phase return false. check dependences of [" + mod->name() + "]");
  }
  for (auto &mod : modules)
    ok &= mod->init();
  return ok;
}
#include "InputManager.h"
bool AnalysisManager::run(int) {
  // load number of events
  auto GlobalOption = [this]() -> const OptionManager * {
    for (auto &mod : modules)
      if (mod->name() == "InputManager")
        return static_cast<const InputManager *>(mod.get())->GlobalOption();
    return nullptr;
  };
  bool ok = true;
  unsigned long long N_event = 1;
  if (GlobalOption()->has("repeat"))
    N_event = GlobalOption()->get<double>("repeat");
  for (unsigned long long event = 0; event < N_event; ++event) {
    for (auto &mod : modules) {
      ok &= mod->run(event);
      if (!ok)
        break;
    }
  }
  return ok;
}
bool AnalysisManager::finish() {
  bool ok = true;
  for (auto &mod : modules)
    ok &= mod->finish();
  for (auto &mod : modules)
    ok &= mod->postfinish();
  return ok;
}

bool AnalysisManager::registerModule(std::unique_ptr<Module> module) {
  std::cout << "[" << module->name() << "](" << module->list() << ") registered" << std::endl;
  modules.push_back(std::move(module));
  return true;
}
bool AnalysisManager::hasModule(const std::string &name) const {
  for (auto &mod : modules)
    if (mod->name() == name)
      return true;
  return false;
}
