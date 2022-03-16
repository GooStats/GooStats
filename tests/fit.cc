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
  double fit(int argc, const char *argv[]) {
    auto ana = std::unique_ptr<AnalysisManager>(new AnalysisManager{});

    auto inputManager = std::unique_ptr<InputManager>(new InputManager(argc, argv));
    inputManager->setInputBuilder(new SimpleInputBuilder());
    auto outManager = std::unique_ptr<OutputManager>(new OutputManager());
    auto out_ptr = outManager.get();
    outManager->setOutputBuilder(new SimpleOutputBuilder());
    outManager->setPlotManager(new SimplePlotManager());

    StatModule::setup(inputManager.get());
    auto gsFit = std::unique_ptr<GSFitManager>(new GSFitManager{});
    StatModule::setup(gsFit.get());
    StatModule::setup(outManager.get());

    ana->registerModule(std::move(inputManager));
    ana->registerModule(std::unique_ptr<PrepareData>(new PrepareData()));
    ana->registerModule(std::unique_ptr<SimpleFit>(new SimpleFit()));
    ana->registerModule(std::move(outManager));

    ana->init();
    ana->run();
    ana->finish();

    return out_ptr->getOutputHelper()->value("likelihood");
  }
}  // namespace GooStats
