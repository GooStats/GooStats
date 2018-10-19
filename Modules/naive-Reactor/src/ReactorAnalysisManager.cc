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
#include "SumPdf.h"
#include "InputManager.h"
#include "TRandom.h"
#include "SpectrumBuilder.h"
#include "OutputManager.h"
#include "BatchOutputManager.h"
bool ReactorAnalysisManager::run(int) {
  auto gOp = inputManager->GlobalOption();
  long long seed;
  if(gOp->has("seed"))
    seed = ::atoi(gOp->query("seed").c_str());
  else
    seed = time(nullptr);
  gRandom->SetSeed(seed);
  if(gOp->hasAndYes("fitFakeData")) { // fit randomly generated data
    if(gOp->hasAndYes("fitRealDataFirst")) {
      findModule("GSFitManager")->run(-1);
      static_cast<OutputManager*>(findModule("OutputManager"))->subFit(-1);
    }
    if(gOp->hasAndYes("fitAsimov")) {
      inputManager->fillAsimovData();
      if(gOp->hasAndYes("fitInverseMH")) {
	auto deltaM2s = inputManager->Datasets().front()->get<std::vector<Variable*>>("deltaM2s");
	deltaM2s[1]->value = - deltaM2s[1]->value;
	deltaM2s[1]->lowerlimit = - deltaM2s[1]->upperlimit;
	deltaM2s[1]->upperlimit = - deltaM2s[1]->lowerlimit;
	this->findModule("GSFitManager")->run(0);
	static_cast<OutputManager*>(this->findModule("OutputManager"))->subFit(0);
	deltaM2s[1]->value = - deltaM2s[1]->value;
	deltaM2s[1]->lowerlimit = - deltaM2s[1]->upperlimit;
	deltaM2s[1]->upperlimit = - deltaM2s[1]->lowerlimit;
      }
      this->AnalysisManager::run();
    } else {
      int N = gOp->has("repeat")?::atoi(gOp->query("repeat").c_str()):1;
      for(int i = 0;i<N;++i) {
	inputManager->resetPars();
	gRandom->SetSeed(seed+i);
	inputManager->fillRandomData();
	if(gOp->hasAndYes("fitInverseMH")) {
	  auto deltaM2s = inputManager->Datasets().front()->get<std::vector<Variable*>>("deltaM2s");
	  deltaM2s[1]->value = - deltaM2s[1]->value;
	  deltaM2s[1]->lowerlimit = - deltaM2s[1]->upperlimit;
	  deltaM2s[1]->upperlimit = - deltaM2s[1]->lowerlimit;
	  this->findModule("GSFitManager")->run(i);
	  static_cast<OutputManager*>(this->findModule("OutputManager"))->subFit(i);
	  deltaM2s[1]->value = - deltaM2s[1]->value;
	  deltaM2s[1]->lowerlimit = - deltaM2s[1]->upperlimit;
	  deltaM2s[1]->upperlimit = - deltaM2s[1]->lowerlimit;
	}
	this->AnalysisManager::run(i);
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
    if(gOp->hasAndYes("getDeltaChisquare")) {
      auto deltaM2s = inputManager->Datasets().front()->get<std::vector<Variable*>>("deltaM2s");
      deltaM2s[1]->value = - deltaM2s[1]->value;
      deltaM2s[1]->lowerlimit = - deltaM2s[1]->upperlimit;
      deltaM2s[1]->upperlimit = - deltaM2s[1]->lowerlimit;
      this->findModule("GSFitManager")->run();
      static_cast<OutputManager*>(this->findModule("OutputManager"))->getBatchOutputManager()->fill_rates();
    }
  }
  return true;
}
bool ReactorAnalysisManager::finish() {
  this->AnalysisManager::finish();
  return true;
}
