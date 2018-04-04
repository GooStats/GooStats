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
#include "Bit.h"
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
    void setInputManager(InputManager *);
    void setOutputManager(OutputManager *);
    static Bit getBit();
    virtual bool init();
    virtual bool run();
    virtual bool finish();
    GooPdf *get_sumpdf() { return sumpdf; }
    bool checkGPU() const;
  protected:
    GooPdf *sumpdf = nullptr;
    std::list<std::shared_ptr<Module> > m_modules;
    std::shared_ptr<FitManager> fitManager;
    std::shared_ptr<InputManager> inputManager;
    std::shared_ptr<OutputManager> outputManager;
};
#define information() Form("%s : line %d",__FILE__, __LINE__)
#endif
