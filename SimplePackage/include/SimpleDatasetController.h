/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef SimpleDatasetControllers_H
#define SimpleDatasetControllers_H
/*! \class SimpleDatasetController
 *  \brief Manager class of dataset, the basic unit exposed to GooFit. A dataset
 *  correspond to one piece of likelihood. It can be more specific dataset: a
 *  specrtrumdataset, with spectrum, components, exposure etc. or a pull-tem
 *  dataset. A pull term data set can be of rate or generally on anything, or a
 *  pull term on the relationship between terms.
 *
 *  The datasetmanager is desgined to take observer pattern, and usually
 *  multiple datasetmanager will listen to one common configsetmanager.
 */
#include "DatasetController.h"
class SimpleDatasetController : public DatasetController {
  public:
  /// @param _c ConfigsetManger storing options
  /// @param n name of the controller, also name of the DatasetManger to be created
  explicit SimpleDatasetController(ConfigsetManager *_c, const std::string &n = "main") : DatasetController(_c, n){};
  bool collectInputs(DatasetManager *) override;
  bool buildLikelihood(DatasetManager *) override;
};
#endif
