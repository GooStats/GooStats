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

int main (int argc, char** argv) {
  AnalysisManager *ana = new ReactorAnalysisManager();
  InputManager *inputManager = new InputManager(argc,argv);
  InputBuilder *builder = new ReactorInputBuilder();
  builder->installSpectrumBuilder(new ReactorSpectrumBuilder());
  inputManager->setInputBuilder(builder);
  ana->setInputManager(inputManager);
  OutputManager *outManager = new OutputManager();
  outManager->setOutputBuilder(new SimpleOutputBuilder());
  outManager->setPlotManager(new SimplePlotManager());
  ana->setOutputManager(outManager);

  ana->init();
  ana->run();
  ana->finish();

  return 0;
}
