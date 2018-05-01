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
DatasetManager *PullDatasetController::createDataset() {
  return new DatasetManager(name);
}
#include "ConfigsetManager.h"
#include "GooStatsException.h"
bool PullDatasetController::collectInputs(DatasetManager *dataset) {
  try {
    dataset->set("var", configset->var(dataset->name()));
    dataset->set("exposure", ::atof(configset->query("exposure").c_str()));
    dataset->set("mean", ::atof(configset->query(dataset->name()+"_centroid").c_str()));
    dataset->set("sigma", ::atof(configset->query(dataset->name()+"_sigma").c_str()));
  } catch (GooStatsException &ex) {
    std::cout<<"Exception caught during fetching parameter configurations. probably you missed iterms in your configuration files. Read the READ me to find more details"<<std::endl;
    std::cout<<"If you think this is a bug, please email to Xuefeng Ding<xuefeng.ding.physics@gmail.com> or open an issue on github"<<std::endl;
    throw ex;
  }
  return true; 
}
bool PullDatasetController::configureParameters(DatasetManager *) {
  return true; 
};
#include "PullPdf.h"
bool PullDatasetController::buildLikelihoods(DatasetManager *dataset) {
  GooPdf *pdf = new PullPdf(dataset->name(),
      dataset->get<Variable*>("var"),
      dataset->get<double>("mean"),
      dataset->get<double>("sigma"),
      dataset->get<double>("exposure"));
  this->setLikelihood(dataset,pdf);
  return true; 
};
