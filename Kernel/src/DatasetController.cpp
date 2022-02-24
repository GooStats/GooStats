//
// Created by Xuefeng Ding on 2/11/22.
//

#include "DatasetController.h"
#include "ConfigsetManager.h"
DatasetManager *DatasetController::createDataset() {
  auto dataset = new DatasetManager(name, configset->name());
  dataset->setController(this);
  return dataset;
}