/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "solarB8InputBuilder.h"
#include "solarB8DatasetController.h"
#include "Utility.h"
#include "ConfigsetManager.h"
#include "PullDatasetController.h"
std::vector<std::shared_ptr<DatasetController>> solarB8InputBuilder::buildDatasetsControllers(ConfigsetManager *configset) {
  auto controllers = this->SimpleInputBuilder::buildDatasetsControllers(configset);
  controllers[0]=std::shared_ptr<DatasetController>(new solarB8DatasetController(configset));
  return controllers;
}
