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
#include <memory>

#include "goofit/PDFs/ExpPdf.hh"
#include "NewExpPdf.hh"
#include "ProductPdf.h"
#include "SumPdf.h"
#include "TH1D.h"
#include "TF1.h"
#include "InputManager.h"
#include "TRandom.h"
#include "SpectrumBuilder.h"
bool ReactorAnalysisManager::run() {
  auto gOp = inputManager->GlobalOption();
  if(gOp->hasAndYes("fitFakeData")) { // fit randomly generated data
    int N = gOp->has("repeat")?::atoi(gOp->query("repeat").c_str()):1;
    long long seed;
    if(this->inputManager->Configsets().front()->has("seed"))
      seed = ::atoi(this->inputManager->Configsets().front()->query("seed").c_str());
    else
      seed = time(nullptr);
    for(int i = 0;i<N;++i) {
      gRandom->SetSeed(seed+i);
      inputManager->resetPars();
      inputManager->fillRandomData();
      if(gOp->hasAndYes("fitInverseMH")) {
	auto deltaM2s = inputManager->Datasets().front()->get<std::vector<Variable*>>("deltaM2s");
	deltaM2s[1]->value = - deltaM2s[1]->value;
	deltaM2s[1]->lowerlimit = - deltaM2s[1]->upperlimit;
	deltaM2s[1]->upperlimit = - deltaM2s[1]->lowerlimit;
	inputManager->cachePars();
      }
      this->AnalysisManager::run();
      if(gOp->hasAndYes("fitInverseMH")) {
	auto deltaM2s = inputManager->Datasets().front()->get<std::vector<Variable*>>("deltaM2s");
	inputManager->resetPars();
	deltaM2s[1]->value = - deltaM2s[1]->value;
	deltaM2s[1]->lowerlimit = - deltaM2s[1]->upperlimit;
	deltaM2s[1]->upperlimit = - deltaM2s[1]->lowerlimit;
	inputManager->cachePars();
      }
    }
  } else { // fitdata
    SpectrumBuilder spcb(inputManager->getProvider());
    for(auto dataset: inputManager->Datasets()) {
      SumPdf *sumpdf = dynamic_cast<SumPdf*>(dataset->getLikelihood());
      if(!sumpdf) continue;
      Variable *Evis = dataset->get<Variable*>("Evis");
      BinnedDataSet *data = spcb.loadRawSpectrum(Evis,"Data");
      sumpdf->setData(data);
    }
    this->AnalysisManager::run();
    auto deltaM2s = inputManager->Datasets().front()->get<std::vector<Variable*>>("deltaM2s");
    inputManager->resetPars();
    deltaM2s[1]->value = - deltaM2s[1]->value;
    deltaM2s[1]->lowerlimit = - deltaM2s[1]->upperlimit;
    deltaM2s[1]->upperlimit = - deltaM2s[1]->lowerlimit;
    inputManager->cachePars();
    this->AnalysisManager::run();
  }
  return true;
}
bool ReactorAnalysisManager::finish() {
  this->AnalysisManager::finish();
  return true;
}
