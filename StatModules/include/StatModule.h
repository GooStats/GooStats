/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef StatModule_H
#define StatModule_H
// interface for the statistical analysis unit of GooStats Analysis program
#include "Module.h"
class InputManager;
class GSFitManager;
class OutputManager;
class OptionManager;
class StatModule : public Module {
  public:
    StatModule(const std::string &_name) : Module(_name) { }
    virtual ~StatModule() { }
  public:
    InputManager *getInputManager();
    const InputManager *getInputManager() const;
    GSFitManager *getGSFitManager();
    const GSFitManager *getGSFitManager() const;
    OutputManager *getOutputManager();
    const OptionManager *GlobalOption() const;
    static void setup(Module *obj) { infoHolder.registerDependence(obj); }
  protected:
    static Module infoHolder;
};
#endif
