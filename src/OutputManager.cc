/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "OutputManager.h"
#include "BatchOutputManager.h"
#include "InputManager.h"
#include "OutputBuilder.h"
#include "OutputHelper.h"
#include <iostream>
#include "GooStatsException.h"
#include "PlotManager.h"
void OutputManager::setBatchOutputManager(BatchOutputManager *b) {
  batchOut = std::shared_ptr<BatchOutputManager>(b);
}
void OutputManager::setOutputBuilder(OutputBuilder *o) {
  outputBuilder = std::shared_ptr<OutputBuilder>(o);
}
void OutputManager::setPlotManager(PlotManager *p) {
  plot = std::shared_ptr<PlotManager>(p);
}
bool OutputManager::init() {
  if(!batchOut) {
    std::cout<<"Warning: BatchOutputManager not set, default are used."<<std::endl;
    setBatchOutputManager(new BatchOutputManager());
  }
  if(!outputBuilder) {
    std::cout<<"Error: OutputBuilder not set, please set it before calling OutputManager::init"<<std::endl;
    throw GooStatsException("OutputManager not set in OutputManager");
  }
  if(!plot) {
    std::cout<<"Error: PlotManager not set, please set it before calling OutputManager::init"<<std::endl;
    throw GooStatsException("PlotManager not set in OutputManager");
  }
  outputHelper = std::make_shared<OutputHelper>();

  batchOut->setOutputFileName(inputManager->getOutputFileName());
  batchOut->init();
  batchOut->bindAllParameters(inputManager->getTotalPdf());
  outputBuilder->registerOutputTerms(outputHelper.get(),inputManager);
  outputBuilder->bindAllParameters(batchOut.get(),outputHelper.get());

  plot = std::make_shared<PlotManager>();
  plot->setOutputFileName(inputManager->getOutputFileName());
  plot->init();
  return true;
}
bool OutputManager::run() {
  outputHelper->flush(inputManager);
  outputBuilder->flushOstream(batchOut.get(),outputHelper.get(),std::cout);
  outputBuilder->draw(plot.get(),inputManager);
  batchOut->run();
  return true;
}
bool OutputManager::finish() {
  batchOut->finish();
  return true;
}
