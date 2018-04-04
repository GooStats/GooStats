/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "JunoDatasetController.h"
#include "ConfigsetManager.h"
#include "GooStatsException.h"
bool JunoDatasetController::collectInputs(DatasetManager *dataset) {
  bool ok = this->SimpleDatasetController::collectInputs(dataset);
  if(!ok) return false;
  try {
    std::vector<Variable*> fractions;
    Variable *U235 = configset->createVar("U235",
	::atof(configset->query("U235_init").c_str()),
	::atof(configset->query("U235_err").c_str()),
	::atof(configset->query("U235_min").c_str()),
	::atof(configset->query("U235_max").c_str()));
    fractions.push_back(U235);
    Variable *U238 = configset->createVar("U238",
	::atof(configset->query("U238_init").c_str()),
	::atof(configset->query("U238_err").c_str()),
	::atof(configset->query("U238_min").c_str()),
	::atof(configset->query("U238_max").c_str()));
    fractions.push_back(U238);
    Variable *Pu239 = configset->createVar("Pu239",
	::atof(configset->query("Pu239_init").c_str()),
	::atof(configset->query("Pu239_err").c_str()),
	::atof(configset->query("Pu239_min").c_str()),
	::atof(configset->query("Pu239_max").c_str()));
    fractions.push_back(Pu239);
    Variable *Pu241 = configset->createVar("U241",
	::atof(configset->query("Pu241_init").c_str()),
	::atof(configset->query("Pu241_err").c_str()),
	::atof(configset->query("Pu241_min").c_str()),
	::atof(configset->query("Pu241_max").c_str()));
    fractions.push_back(Pu241);
    dataset->set("fractions", fractions);
    std::vector<double> coefficients;
    coefficients.push_back(::atof(configset->query("Huber_U235_0").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_U235_1").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_U235_2").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_U238_0").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_U238_1").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_U238_2").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_U235_0").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_U235_1").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_U235_2").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_Pu241_0").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_Pu241_1").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_Pu241_2").c_str()));
    dataset->set("coefficients", coefficients);
  } catch (GooStatsException &ex) {
    std::cout<<"Exception caught during fetching parameter configurations. probably you missed iterms in your configuration files. Read the READ me to find more details"<<std::endl;
    std::cout<<"If you think this is a bug, please email to Xuefeng Ding<xuefeng.ding.physics@gmail.com> or open an issue on github"<<std::endl;
    throw ex;
  }
  return true; 
}
