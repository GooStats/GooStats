/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "JunoInputBuilder.h"
#include "JunoDatasetController.h"
std::vector<std::shared_ptr<DatasetController>> JunoInputBuilder::buildDatasetsControllers(ConfigsetManager *configset) {
  std::vector<std::shared_ptr<DatasetController>> controllers;
  controllers.push_back(std::shared_ptr<DatasetController>(new JunoDatasetController(configset)));
  return controllers;
}
