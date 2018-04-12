/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ReactorInputBuilder.h"
#include "ReactorDatasetController.h"
std::vector<std::shared_ptr<DatasetController>> ReactorInputBuilder::buildDatasetsControllers(ConfigsetManager *configset) {
  std::vector<std::shared_ptr<DatasetController>> controllers;
  controllers.push_back(std::shared_ptr<DatasetController>(new ReactorDatasetController(configset)));
  return controllers;
}
