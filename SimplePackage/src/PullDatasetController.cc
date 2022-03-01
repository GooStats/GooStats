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
#include "RawSpectrumProvider.h"
#include "goofit/Variable.h"
bool PullDatasetController::collectInputs(DatasetManager *dataset) {
  try {
    const auto &varName = dataset->name().substr(0, dataset->name().size() - 5);// remove _pull
    dataset->set("varName", varName);
    auto var = configset->var(varName);
    dataset->set("var", var);
    std::string type = configset->has(varName + "_pullType") ? configset->get(varName + "_pullType") : "gaus";
    dataset->set("type", type);
    auto expoName = "exposure." + varName + "_pull";
    if (configset->has(expoName) || configset->has<double>(expoName)) {
      dataset->set("exposure", configset->getOrConvert(expoName));
    } else {
      std::cerr << "Warning: for compatibility, use total exposure in Configset [" << configset->name()
                << "] for pull [" << varName << "]" << std::endl;
      dataset->set("exposure", configset->getOrConvert("exposure"));
    }
    if (type == "gaus") {
      dataset->set("mean", configset->getOrConvert(varName + "_centroid"));
      dataset->set("sigma", configset->getOrConvert(varName + "_sigma"));
      dataset->set("half", configset->hasAndYes(varName + "_half"));
    } else if (type == "poisson") {
      dataset->set("eff", configset->var(varName + "_eff"));
      dataset->set("mu_sig", configset->getOrConvert(varName + "_sig"));
      dataset->set("mu_bkg", configset->getOrConvert(varName + "_bkg"));
    } else {
      throw GooStatsException("Unknown Pull type: [" + configset->get(varName + "_pullType") + "]");
    }
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
#include "PoissonPullPdf.h"
#include "PullPdf.h"
bool PullDatasetController::buildLikelihood(DatasetManager *dataset) {
  auto type = dataset->get<std::string>("type");
  if (type == "gaus") {
    std::cout << "Creating gauss pull [" << dataset->get<std::string>("varName") << "] with exposure ["
              << dataset->get<double>("exposure") << "] and par (" << dataset->get<double>("mean") << ") ± ("
              << dataset->get<double>("sigma") << ") " << std::endl;
    GooPdf *pdf =
            new PullPdf(dataset->fullName(), dataset->get<Variable *>("var"), dataset->get<double>("mean"),
                        dataset->get<double>("sigma"), dataset->get<double>("exposure"), dataset->get<bool>("half"));
    dataset->setLikelihood(pdf);
  } else if (type == "poisson") {
    std::cout << "Creating poisson pull [" << dataset->get<std::string>("varName") << "] with exposure ["
              << dataset->get<double>("exposure") << "] and par S (" << dataset->get<double>("mu_sig") << ") B ("
              << dataset->get<double>("mu_bkg") << ") eff (" << dataset->get<Variable *>("eff")->value << ")"
              << std::endl;
    GooPdf *pdf = new PoissonPullPdf(dataset->fullName(), dataset->get<Variable *>("var"),
                                     dataset->get<Variable *>("eff"), dataset->get<double>("exposure"),
                                     dataset->get<double>("mu_sig"), dataset->get<double>("mu_bkg"));
    dataset->setLikelihood(pdf);
  } else {
    throw GooStatsException("Unknown Pull type: [" + type + "]");
  }
  return true;
}
