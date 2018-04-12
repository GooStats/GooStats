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
bool ReactorAnalysisManager::init() {
  this->AnalysisManager::init();

 inputManager->fillRandomData();
 // Examples about how to do CPU version fit
 //DatasetManager *dataset = *inputManager->Datasets().begin();
 // Variable *Evis = dataset->get<Variable*>("Evis");
// auto data = std::move(datas.begin()->second);
// TH1* data_h = new TH1D("h","h",Evis->numbins,Evis->lowerlimit,Evis->upperlimit);
// for(int i = 0;i<Evis->numbins;++i) {
//   data_h->SetBinContent(i,data[i*3+1]);
// }
//
// std::cout<<"CPU version: -------------------------------------------------"<<std::endl;
// TF1 *f = new TF1("f","exp([0]*x+[1]*x*x)*[2]",1,8);
// f->SetParameters(-0.2,-0.025,500);
// data_h->Fit("f","M");
// std::cout<<"CPU version: -------------------------------------------------"<<std::endl;

  return true;
}
bool ReactorAnalysisManager::finish() {
  this->AnalysisManager::finish();
  return true;
}
