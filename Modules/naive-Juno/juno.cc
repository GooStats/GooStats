/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "goofit/FitManager.h"
#include <iostream>
#include <sys/time.h>
#include <sys/times.h>
#include "JunoAnalysisManager.h"
#include "InputManager.h"
#include "JunoInputBuilder.h"
#include "JunoSpectrumBuilder.h"
#include "OutputManager.h"
#include "SimpleOutputBuilder.h"
#include "SimplePlotManager.h"

int main (int argc, char** argv) {
  AnalysisManager *ana = new JunoAnalysisManager();
  InputManager *inputManager = new InputManager(argc,argv);
  InputBuilder *builder = new JunoInputBuilder();
  builder->installSpectrumBuilder(new JunoSpectrumBuilder());
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
