/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "PrepareData.h"

#include "GSFitManager.h"
#include "InputManager.h"
#include "MultiComponentDatasetController.h"
#include "OutputManager.h"
#include "SimpleSpectrumBuilder.h"
#include "TRandom.h"
#include "goofit/PDFs/SumPdf.h"
bool PrepareData::init() {
  if (GlobalOption()->has("seed"))
    seed = GlobalOption()->getOrConvert<int>("seed");
  else
    seed = time(nullptr);
  gRandom->SetSeed(seed);
  return true;
}
bool PrepareData::run(int ev) {
  if (GlobalOption()->hasAndYes("fitFakeData")) {  // fit randomly generated data
    if (GlobalOption()->hasAndYes("fitRealDataFirst") && ev == 0) {
      getGSFitManager()->run(-1);      // -1 = don't save
      getOutputManager()->subFit(-1);  // -1 = don't save
    } else
      getInputManager()->resetPars();
    if (GlobalOption()->hasAndYes("fitAsimov")) {
      getInputManager()->fillAsimovData();
    } else {
      gRandom->SetSeed(seed + ev);
      getInputManager()->fillRandomData();
    }
  } else {  // fitdata
    if (ev > 0)
      throw GooStatsException("Why you want to fit same dataset several times?");
    //    SimpleSpectrumBuilder spcb(getInputManager()->getProvider());
    for (auto controller : getInputManager()->Controllers()) {
      if (dynamic_cast<MultiComponentDatasetController*>(controller) != nullptr) {
        auto dataset = controller->getDataset();
        auto sumpdf = dataset->getLikelihood();
        sumpdf->setData(dataset->get<BinnedDataSet*>("data"));
      }
    }
  }
  return true;
}
