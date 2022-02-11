//
// Created by Xuefeng Ding on 2/11/22.
//

#include "DatasetController.h"
DatasetManager *DatasetController::createDataset() {
  auto dataset = new DatasetManager(name);
  dataset->setController(this);
  return dataset;
}