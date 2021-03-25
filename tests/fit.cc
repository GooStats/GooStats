/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "gtest/gtest.h"
#include "AnalysisManager.h"
#include "InputManager.h"
#include "SimpleInputBuilder.h"
#include "OutputManager.h"
#include "SimpleOutputBuilder.h"
#include "SimplePlotManager.h"
#include "GSFitManager.h"
#include "PrepareData.h"
#include "SimpleFit.h"
#include "OutputHelper.h"

TEST (GooStats, simpleFit) {
  AnalysisManager *ana = new AnalysisManager();

  const char *argv[2] = {"exe","toyMC.cfg"};
  auto inputManager = new InputManager(2,argv);
  inputManager->setInputBuilder(new SimpleInputBuilder());
  auto outManager = new OutputManager();
  outManager->setOutputBuilder(new SimpleOutputBuilder());
  outManager->setPlotManager(new SimplePlotManager());

  StatModule::setup(inputManager);
  StatModule::setup(new GSFitManager());
  StatModule::setup(outManager);

  ana->registerModule(inputManager);
  ana->registerModule(new PrepareData());
  ana->registerModule(new SimpleFit());
  ana->registerModule(outManager);

  ana->init();
  ana->run();
  ana->finish();

  auto outputHelper = outManager->getOutputHelper();

  EXPECT_NEAR(outputHelper->value("likelihood"),357.95898,0.00001);
}
