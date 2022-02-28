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
#include "DatasetManager.h"
#include "GooStatsException.h"
#include "goofit/Variable.h"
#include "RawSpectrumProvider.h"
bool PullDatasetController::collectInputs(DatasetManager *dataset) {
  try {
    const auto &varName = dataset->name().substr(0, dataset->name().size() - 5);// remove _pull
    auto var = configset->var(varName);
    if (!configset->has(varName + "_pullType") || configset->get(varName + "_pullType") == "gaus") {// default: gaus
      dataset->set("type", std::string("gaus"));
      auto expoName = "exposure." + varName + "_pull";
      if (configset->has(expoName)) {
        dataset->set("exposure", configset->getOrConvert(expoName));
      } else {
        std::cerr << "Warning: for compatibility, use total exposure in Configset [" << configset->name()
                  << "] for pull [" << varName << "]" << std::endl;
        dataset->set("exposure", configset->getOrConvert("exposure"));
      }
      dataset->set("mean", configset->getOrConvert(varName + "_centroid"));
      dataset->set("sigma", configset->getOrConvert(varName + "_sigma"));
      dataset->set("half", configset->hasAndYes(varName + "_half"));
    } else {
      throw GooStatsException("Unknown Pull type: [" + configset->get(varName + "_pullType") + "]");
    }
    dataset->set("var", var);
  } catch (GooStatsException &ex) {
    std::cout << "Exception caught during fetching parameter configurations. probably you missed iterms in your "
                 "configuration files. Read the READ me to find more details"
              << std::endl;
    std::cout << "If you think this is a bug, please email to Xuefeng Ding<xuefeng.ding.physics@gmail.com> or open an "
                 "issue on github"
              << std::endl;
    throw ex;
  }
  return true;
}
#include "PullPdf.h"
bool PullDatasetController::buildLikelihood(DatasetManager *dataset) {
  auto type = dataset->get<std::string>("type");
  if (type == "gaus") {
    GooPdf *pdf =
            new PullPdf(dataset->fullName(), dataset->get<Variable *>("var"), dataset->get<double>("mean"),
                        dataset->get<double>("sigma"), dataset->get<double>("exposure"), dataset->get<bool>("half"));
    dataset->setLikelihood(pdf);
  } else {
    throw GooStatsException("Unknown Pull type: [" + type + "]");
  }
  return true;
}
