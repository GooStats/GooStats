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
#include "goofit/Variable.h"
#include "GooStatsException.h"
#include "TextOutputManager.h"
#include "OutputManager.h"
#include "TFile.h"
#include "Utility.h"
#include "SumLikelihoodPdf.h"
#include "OptionManager.h"
#include "InputManager.h"
bool BatchOutputManager::init() {
  const InputManager *inputManager = static_cast<InputManager*>(find("InputManager"));
  const OptionManager *gOp = inputManager->GlobalOption();
  if(gOp->has("unit")) TextOutputManager::set_unit(gOp->get("unit"));
  cd();
  tree = std::make_shared<TTree>("fit_results","Fit result of GooStats"); 
  bindAllParameters(inputManager->getTotalPdf());
  return true;
}
bool BatchOutputManager::run(int ) {
  tree->Fill();
  nfit = 0;
  return true;
}
bool BatchOutputManager::finish() {
  flush_tree();
  return true;
}
void BatchOutputManager::fill_rates() {
  if(pdf) {
    if(nfit>=200) {
      throw GooStatsException("Not enough space for subtest reserved in BatchOutputManager::fit_results");
    }
    pdf->getParameters(vars);
    for(auto var : vars) {
      fill(var->name,var->value);
      fill(var->name+"_err",var->error);
    }
    ++nfit;
  }
}
void BatchOutputManager::cd() {
  static_cast<OutputManager*>(find("OutputManager"))->getOutputFile()->cd();
}
void BatchOutputManager::flush_tree() {
  cd();
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
      OPRINTF(" %s\n", TextOutputManager::rate(var->name,var->value,var->error,var->upperlimit,var->lowerlimit,var->apply_penalty,var->penalty_mean,var->penalty_sigma).c_str());
    else
      OPRINTF(" %s\n", TextOutputManager::qch(var->name,var->value,var->error,var->upperlimit,var->lowerlimit,var->apply_penalty,var->penalty_mean,var->penalty_sigma).c_str());
  }
  OPRINTF("\n");
  OPRINTF(" chi^2                             = %.1lf\n",        goodness["chi2"]);
  OPRINTF(" chi^2/N-DOF                       = %.4lf\n",        goodness["chi2/NDF"]);
  OPRINTF(" p-value                           = %.3lf\n",        goodness["p-value"]);
  OPRINTF(" Minimized -2Ln(Likelihood)        = %.2lf\n",        goodness["likelihood"]*2);
  OPRINTF(" Likelihood p-value                = %.3lf ± %.3lf\n",goodness["LLp"],goodness["LLpErr"]);
}
void BatchOutputManager::bindAllParameters(const SumLikelihoodPdf *pdf_) {
  pdf = pdf_;
  pdf->getParameters(vars);
  tree->Branch("nfit",&nfit,"nfit/I");
  for(auto var : vars) {
    bind(var->name);
    bind(var->name+"_err");
  }
}
void BatchOutputManager::bind(const std::string &brName) {
  tree->Branch(GooStats::Utility::escape(brName).c_str(), 
      results[brName], 
      (GooStats::Utility::escape(brName)+"[nfit]/D").c_str());
}
