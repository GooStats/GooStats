/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
/*! \class ConfigsetManager
 *  \brief Manager class of configset, the basic unit of dataset in GooStats.
 *  A configset correspond to a configuration set, that is one single
 *  configuration file. It may contain mutiple spectra plus pull terms, and we
 *  call each of them Dataset in GooStats
 *
 *   This class will collect raw spectrum, parse configurations, and as a
 *   controller, it will setup parameters and build the objects recognizable for
 *   each Dataset.
 */
#ifndef ConfigsetManagers_H
#define ConfigsetManagers_H
#include <memory>
#include <vector>

#include "BasicManager.h"
#include "DatasetController.h"
#include "OptionManager.h"
class ConfigsetManager : public BasicManager, public OptionManager {
 public:
  explicit ConfigsetManager(const std::string &name_) : BasicManager(name_) {}
  virtual ~ConfigsetManager() = default;
};
#endif
