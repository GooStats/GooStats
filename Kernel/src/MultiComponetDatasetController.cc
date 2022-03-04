/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Institute: Department of Physics, Princeton University
// Date: 3/2/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/

#include "MultiComponentDatasetController.h"
#include "Utility.h"

std::vector<std::string> MultiComponentDatasetController::getComponents(DatasetManager *dataset) const {
  return dataset->get<std::vector<std::string>>("components");
}
