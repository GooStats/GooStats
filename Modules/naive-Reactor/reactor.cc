/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ReactorAnalysisManager.h"
#include "InputManager.h"
#include "ReactorInputBuilder.h"
#include "ReactorSpectrumBuilder.h"
#include "OutputManager.h"
#include "SimpleOutputBuilder.h"
#include "SimplePlotManager.h"
#include "GSFitManager.h"
#include "ContourManager.h"
#include "CorrelationManager.h"

int main (int argc, char** argv) {
  AnalysisManager *ana = new ReactorAnalysisManager();
  InputManager *inputManager = new InputManager(argc,argv);
  InputBuilder *builder = new ReactorInputBuilder();
  builder->installSpectrumBuilder(new ReactorSpectrumBuilder());
  inputManager->setInputBuilder(builder);
  ana->setInputManager(inputManager);
  ana->registerModule(inputManager);

  GSFitManager *gsFitManager = new GSFitManager();
  gsFitManager->registerDependence(inputManager);
  ana->registerModule(gsFitManager);

  OutputManager *outManager = new OutputManager();
  outManager->registerDependence(inputManager);
  outManager->registerDependence(gsFitManager);
  outManager->setOutputBuilder(new SimpleOutputBuilder());
  ana->setOutputManager(outManager);
  ana->registerModule(outManager);

  PlotManager *plotManager = new SimplePlotManager();
  plotManager->registerDependence(gsFitManager);
  plotManager->registerDependence(outManager);
  plotManager->registerDependence(inputManager);
  outManager->setPlotManager(plotManager);

  CorrelationManager *correlationManager = new CorrelationManager();
  correlationManager->registerDependence(inputManager);
  correlationManager->registerDependence(gsFitManager);
  correlationManager->registerDependence(outManager);
  ana->registerModule(correlationManager);

  ContourManager *contourManager = new ContourManager();
  contourManager->registerDependence(inputManager);
  contourManager->registerDependence(gsFitManager);
  contourManager->registerDependence(outManager);
  ana->registerModule(contourManager);

  ana->init();
  ana->run();
  ana->finish();

  return 0;
}
