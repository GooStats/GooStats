//
// Created by Xuefeng Ding on 2/11/22.
//

#include "DatasetController.h"
#include "ConfigsetManager.h"
DatasetManager *DatasetController::createDataset() {
  dataset = std::make_shared<DatasetManager>(name, configset->name());
  dataset->setController(this);
  return dataset.get();
}