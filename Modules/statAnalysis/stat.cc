/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "AnalysisManager.h"
#include "InputManager.h"
#include "SimpleInputBuilder.h"
#include "OutputManager.h"
#include "SimpleOutputBuilder.h"
#include "SimplePlotManager.h"
#include "GSFitManager.h"
#include "PrepareData.h"
#include "SimpleFit.h"
#include "DiscoveryTest.h"

int main (int argc, const char** argv) {
  AnalysisManager *ana = new AnalysisManager();

  InputManager *inputManager = new InputManager(argc,argv);
  inputManager->setInputBuilder(new SimpleInputBuilder());
  GSFitManager *gsFitManager = new GSFitManager();
  OutputManager *outManager = new OutputManager();
  outManager->setOutputBuilder(new SimpleOutputBuilder());
  outManager->setPlotManager(new SimplePlotManager());

  StatModule::setup(inputManager);
  StatModule::setup(gsFitManager);
  StatModule::setup(outManager);

  PrepareData *data = new PrepareData();
  SimpleFit *fit = new SimpleFit();
  DiscoveryTest *discovery = new DiscoveryTest();

  ana->registerModule(inputManager);
  ana->registerModule(data);
  ana->registerModule(fit);
  ana->registerModule(discovery);
  ana->registerModule(outManager);

  ana->init();
  ana->run();
  ana->finish();

  return 0;
}
