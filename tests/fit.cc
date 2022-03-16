/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "fit.h"

#include "AnalysisManager.h"
#include "GSFitManager.h"
#include "InputManager.h"
#include "OutputHelper.h"
#include "OutputManager.h"
#include "PrepareData.h"
#include "SimpleFit.h"
#include "SimpleInputBuilder.h"
#include "SimpleOutputBuilder.h"
#include "SimplePlotManager.h"
namespace GooStats {
  const OutputHelper *fit(int argc, const char *argv[]) {

    auto ana = new AnalysisManager();

    auto inputManager = new InputManager(argc, argv);
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

    return outManager->getOutputHelper();
  }
}  // namespace GooStats
