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
 *   delegate, it will setup parameters and build the objects recognizable for
 *   each Dataset.
 */
#ifndef ConfigsetManagers_H
#define ConfigsetManagers_H
#include "BasicManager.h"
#include "OptionManager.h"
#include "DatasetController.h"
#include <memory>
#include <vector>
class ConfigsetManager : public BasicManager, public OptionManager {
  public:
    ConfigsetManager(const std::string name_,OptionManager *op) : BasicManager(name_),OptionManager(*op) {};
    ConfigsetManager(BasicManager &manager,OptionManager *op) : BasicManager(manager),OptionManager(*op) {};
  public:
    void setDatasetControllers
      (const std::vector<std::shared_ptr<DatasetController>> &_controllers) {
	controllers = _controllers;
      }
    std::vector<std::shared_ptr<DatasetController>> getDatasetControllers() { 
      return controllers; }
  private:
    std::vector<std::shared_ptr<DatasetController>> controllers;
};
#endif
