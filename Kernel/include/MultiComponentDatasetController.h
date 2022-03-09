/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Institute: Department of Physics, Princeton University
// Date: 3/2/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/

#ifndef BX_GOOSTATS_MULTICOMPONENTDATASETCONTROLLER_H
#define BX_GOOSTATS_MULTICOMPONENTDATASETCONTROLLER_H

#include "DatasetController.h"
#include "RawSpectrumProvider.h"
class MultiComponentDatasetController : public DatasetController {
 public:
  /// @param _c ConfigsetManger storing options
  /// @param n name of the controller, also name of the DatasetManger to be created
  explicit MultiComponentDatasetController(ConfigsetManager *_c, const std::string &n = "main")
      : DatasetController(_c, n){};
  std::vector<std::string> getComponents(DatasetManager *dataset) const;
};

#endif  //BX_GOOSTATS_MULTICOMPONENTDATASETCONTROLLER_H
