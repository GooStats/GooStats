/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "BatchOutputManager.h"
#include "goofit/PDFs/GooPdf.h"
#include "goofit/Variable.h"
#include "TFile.h"
#include "GooStatsException.h"
#include "TextOutputManager.h"
bool BatchOutputManager::init() {
  out = TFile::Open((outName+"_tree.root").c_str(),"RECREATE");
  if(!out->IsOpen()) {
    std::cout<<"Cannot create output root file: <"<<outName<<">"<<std::endl;
    throw GooStatsException("Cannot create output root file");
  }
  //tree = std::make_shared<TTree>("fit_result","Fit result of GooStats");
  tree = new TTree("fit_result","Fit result of GooStats");
  return true;
}
bool BatchOutputManager::run() {
  tree->Fill();
  return true;
}
bool BatchOutputManager::finish() {
  flush_tree();
  out->Close();
  return true;
}
void BatchOutputManager::flush_tree() {
  out->cd();
  tree->Write();
}
void BatchOutputManager::flush_txt(std::ostream &out,std::map<std::string,double> &goodness) const {
  char buff[255];
#define OPRINTF(...) \
  do { sprintf(buff,__VA_ARGS__); \
  out<<std::string(buff)<<std::flush; } while(0)
  OPRINTF("\n");
  OPRINTF("FIT PARAMETERS\n");
  for( auto var : vars ) {
    std::string type = var->name.substr(var->name.find(":")+1);
    if(var->numbins <0) // dirty hack
      OPRINTF(" %s\n", TextOutputManager::rate(var->name,var->value,var->error,var->upperlimit,var->lowerlimit,"cpd/ktons",var->apply_penalty,var->penalty_mean,var->penalty_sigma).c_str());
    else
      OPRINTF(" %s\n", TextOutputManager::qch(var->name,var->value,var->error,var->upperlimit,var->lowerlimit,var->apply_penalty,var->penalty_mean,var->penalty_sigma).c_str());
  }
  OPRINTF("\n");
  OPRINTF(" chi^2                             = %.1lf\n",        goodness["chi2"]);
  OPRINTF(" chi^2/N-DOF                       = %.4lf\n",        goodness["chi2/NDF"]);
  OPRINTF(" p-value                           = %.3lf\n",        goodness["p-value"]);
  OPRINTF(" Minimized Likelihood Value        = %.2lf\n",        goodness["likelihood"]);
  OPRINTF(" Likelihood p-value                = %.3lf ± %.3lf\n",goodness["LLp"],goodness["LLpErr"]);
}
void BatchOutputManager::bindAllParameters(GooPdf *pdf) {
  pdf->getParameters(vars);
  for(auto var : vars) {
    bind(var->name, &var->value);
    bind(var->name+"_err", &var->error);
  }
}
