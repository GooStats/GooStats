/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "StatModule.h"

#include "GSFitManager.h"
#include "InputManager.h"
#include "OutputManager.h"

Module StatModule::infoHolder("infoHolder");
InputManager *StatModule::getInputManager() { return static_cast<InputManager *>(infoHolder.find("InputManager")); }
const InputManager *StatModule::getInputManager() const {
  return static_cast<const InputManager *>(infoHolder.find("InputManager"));
}
GSFitManager *StatModule::getGSFitManager() { return static_cast<GSFitManager *>(infoHolder.find("GSFitManager")); }
const GSFitManager *StatModule::getGSFitManager() const {
  return static_cast<const GSFitManager *>(infoHolder.find("GSFitManager"));
}
OutputManager *StatModule::getOutputManager() { return static_cast<OutputManager *>(infoHolder.find("OutputManager")); }
[[deprecated]] const OptionManager *StatModule::GlobalOption() const {
  return getInputManager()->GlobalOption();
}
