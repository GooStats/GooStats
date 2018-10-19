/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ANALYSIS_MANAGER_H
#define ANALYSIS_MANAGER_H
class Module;
#include <memory>
#include <list>
class InputManager;
class OutputManager;
class InputBuilder;
class ParSyncManager;
class GooPdf;
class FitManager;
#include "GPUManager.h"
#include "GooStatsException.h"
class AnalysisManager {
  public:
    bool registerModule(Module *module);
    bool hasModule(const std::string &name) const;
    Module *findModule(const std::string &name);
    void setInputManager(InputManager *);
    void setOutputManager(OutputManager *);
    virtual bool init();
    virtual bool run(int event = 0);
    virtual bool finish();
    bool checkGPU() const;
  protected:
    std::list<std::shared_ptr<Module> > modules;
    std::shared_ptr<InputManager> inputManager;
    std::shared_ptr<OutputManager> outputManager;
};
#define information() Form("%s : line %d",__FILE__, __LINE__)
#endif
