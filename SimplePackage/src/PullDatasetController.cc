/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "PullDatasetController.h"
#include "ConfigsetManager.h"
#include "GooStatsException.h"
bool PullDatasetController::collectInputs(DatasetManager *dataset) {
  try {
    const auto &varName = dataset->name().substr(0,dataset->name().size()-5); // remove _pull
    dataset->set("var", configset->var(varName));
    if(!configset->has(varName+"_pullType")) { // default: gaus
      dataset->set("type", std::string("gaus"));
      dataset->set("exposure", ::atof(configset->query("exposure").c_str()));
      dataset->set("mean", ::atof(configset->query(varName+"_centroid").c_str()));
      dataset->set("sigma", ::atof(configset->query(varName+"_sigma").c_str()));
    } else if(configset->query(varName+"_pullType")=="square") {
      dataset->set("type", std::string("square"));
      dataset->set("lowerlimit", ::atof(configset->query(varName+"_min").c_str()));
      dataset->set("upperlimit", ::atof(configset->query(varName+"_max").c_str()));
    } else {
      throw GooStatsException("Unknown Pull type: ["+configset->query(varName+"_pullType")+"]");
    }
  } catch (GooStatsException &ex) {
    std::cout<<"Exception caught during fetching parameter configurations. probably you missed iterms in your configuration files. Read the READ me to find more details"<<std::endl;
    std::cout<<"If you think this is a bug, please email to Xuefeng Ding<xuefeng.ding.physics@gmail.com> or open an issue on github"<<std::endl;
    throw ex;
  }
  return true; 
}
bool PullDatasetController::configureParameters(DatasetManager *dataset) {
  auto var = dataset->get<Variable*>("var");
  auto type = dataset->get<std::string>("type");
  if(type=="square") {
    var->lowerlimit = dataset->get<double>("lowerlimit");
    var->upperlimit = dataset->get<double>("upperlimit");
  } else if(type=="gaus") {
    var->apply_penalty = true;
    var->penalty_mean = dataset->get<double>("mean");
    var->penalty_sigma = dataset->get<double>("sigma");
  }
  return true; 
};
#include "PullPdf.h"
bool PullDatasetController::buildLikelihoods(DatasetManager *dataset) {
  auto type = dataset->get<std::string>("type");
  if(type=="gaus") {
    GooPdf *pdf = new PullPdf(dataset->name(),
			      dataset->get<Variable*>("var"),
			      dataset->get<double>("mean"),
			      dataset->get<double>("sigma"),
			      dataset->get<double>("exposure"));
    this->setLikelihood(dataset,pdf);
  }
  return true; 
};
