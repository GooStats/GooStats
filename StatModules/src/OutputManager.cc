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
#include "GSFitManager.h"
#include <iostream>
#include "GooStatsException.h"
#include "PlotManager.h"
#include "TFile.h"
void OutputManager::setOutputFile(const std::string &fname) {
  file = TFile::Open((fname+".root").c_str(),"RECREATE");
  if(!file->IsOpen()) {
    std::cout<<"Cannot create output root file: <"<<fname<<">"<<std::endl;
    throw GooStatsException("Cannot create output root file");
  }
}
void OutputManager::setBatchOutputManager(BatchOutputManager *b) {
  batchOut = std::shared_ptr<BatchOutputManager>(b);
}
void OutputManager::setOutputBuilder(OutputBuilder *o) {
  outputBuilder = std::shared_ptr<OutputBuilder>(o);
}
void OutputManager::setPlotManager(PlotManager *p) {
  plot = std::shared_ptr<PlotManager>(p);
}
bool OutputManager::preinit() {
  bool ok = true;
  if(!batchOut) {
    std::cout<<"Warning: BatchOutputManager not set, default are used."<<std::endl;
    BatchOutputManager *batch = new BatchOutputManager();
    batch->registerDependence(getInputManager());
    batch->registerDependence(this);
    setBatchOutputManager(batch);
  }
  ok &= batchOut->preinit();
  if(!outputBuilder) {
    std::cout<<"Error: OutputBuilder not set, please set it before calling OutputManager::init"<<std::endl;
    throw GooStatsException("OutputManager not set in OutputManager");
  }
  if(plot)
    ok &= plot->preinit();
  else
    std::cout<<"Warning: PlotManager not set, no figure output will be created"<<std::endl;
  outputHelper = std::make_shared<OutputHelper>();
  return ok && this->Module::preinit();
}
bool OutputManager::init() {
  this->setOutputFile(getInputManager()->getOutputFileName());

  batchOut->init();
  outputBuilder->registerOutputTerms(outputHelper.get(),getInputManager(),getGSFitManager());
  outputBuilder->bindAllParameters(batchOut.get(),outputHelper.get());

  if(plot) {
    plot->init();
  }
  return true;
}
void OutputManager::subFit(int event) {
  // calculate likelihood etc.
  outputHelper->flush();
  // fill calculated value to BatchOutputManager::results
  outputBuilder->fillAllParameters(batchOut.get(),outputHelper.get());
  batchOut->fill_rates(); // fill_rates will increase nfit, must be the last one
  // print on screen
  if(event>=0) outputBuilder->flushOstream(batchOut.get(),outputHelper.get(),std::cout);
  // make a plot
  if(plot && !((!getInputManager()->GlobalOption()->has("disablePlots") && getInputManager()->GlobalOption()->has("repeat") && std::atoi(getInputManager()->GlobalOption()->query("repeat").c_str())>0)
      ||getInputManager()->GlobalOption()->hasAndYes("disablePlots"))) {
    outputBuilder->draw(event,getGSFitManager(),plot.get(),getInputManager());
  }
}
bool OutputManager::run(int event) {
  // save to disk
  batchOut->run(event);
  if(plot && !((!getInputManager()->GlobalOption()->has("disablePlots") && getInputManager()->GlobalOption()->has("repeat") && std::atoi(getInputManager()->GlobalOption()->query("repeat").c_str())>0)
      ||getInputManager()->GlobalOption()->hasAndYes("disablePlots"))) {
    plot->run(event);
  }
  static int count = 0;
  if(count/100*100==count) {
    file->Delete("fit_results;1");
    file->cd();
    file->Get("fit_results")->Write();
  }
  ++count;
  return true;
}
bool OutputManager::finish() {
  file->Delete("fit_results;1");
  batchOut->finish();
  if(plot) plot->finish();
  return true;
}
bool OutputManager::postfinish() {
  file->Close();
  delete file;
  file = nullptr;
  return true;
}
