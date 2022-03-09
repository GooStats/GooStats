/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef DatasetControllers_H
#define DatasetControllers_H
/*! \class DatasetController
 *  \brief Manager class of dataset, the basic unit exposed to GooFit. A dataset
 *  correspond to one piece of likelihood. It can be more specific dataset: a
 *  specrtrumdataset, with spectrum, components, exposure etc. or a pull-tem
 *  dataset. A pull term data set can be of rate or generally on anything, or a
 *  pull term on the relationship between terms.
 *
 *  The datasetmanager is desgined to take observer pattern, and usually
 *  multiple datasetmanager will listen to one common configsetmanager.
 */
#include <utility>

#include "DatasetManager.h"
class ConfigsetManager;
class RawSpectrumProvider;
class DatasetController {
 public:
  DatasetController(ConfigsetManager *_c, std::string n = "LL") : configset(_c), name(std::move(n)){};
  DatasetController() = delete;
  virtual ~DatasetController() = default;
  virtual DatasetManager *createDataset();
  virtual bool collectInputs() = 0;
  virtual bool buildLikelihood() = 0;
  ConfigsetManager *getConfigset() { return configset; }
  [[nodiscard]] const DatasetManager *getDataset() const { return dataset.get(); }
  DatasetManager *getDataset() { return dataset.get(); }
  [[nodiscard]] const std::string &getName() const { return name; }

 protected:
  ConfigsetManager *const configset;
  std::shared_ptr<DatasetManager> dataset;
  const std::string name;
};
#endif
