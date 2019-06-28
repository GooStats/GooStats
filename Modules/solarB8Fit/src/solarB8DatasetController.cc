/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "solarB8DatasetController.h"
#include "ConfigsetManager.h"
#include "GooStatsException.h"
bool solarB8DatasetController::collectInputs(DatasetManager *dataset) {
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
    Variable *Pu241 = configset->createVar("Pu241",
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
    coefficients.push_back(::atof(configset->query("Huber_Pu239_0").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_Pu239_1").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_Pu239_2").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_Pu241_0").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_Pu241_1").c_str()));
    coefficients.push_back(::atof(configset->query("Huber_Pu241_2").c_str()));
    dataset->set("coefficients", coefficients);
    dataset->set("reactorPower", ::atof(configset->query("reactorPower").c_str()));
    dataset->set("distance", ::atof(configset->query("distance").c_str()));
    dataset->set("NHatomPerkton", ::atof(configset->query("NHatomPerkton").c_str()));
    std::vector<Variable*> sinTheta_2s;
    Variable *sinTheta12_2 = configset->createVar("sinTheta12_2",
	::atof(configset->query("sinTheta12_2_init").c_str()),
	::atof(configset->query("sinTheta12_2_err").c_str()),
	::atof(configset->query("sinTheta12_2_min").c_str()),
	::atof(configset->query("sinTheta12_2_max").c_str()));
    sinTheta_2s.push_back(sinTheta12_2);
    Variable *sinTheta13_2 = configset->createVar("sinTheta13_2",
	::atof(configset->query("sinTheta13_2_init").c_str()),
	::atof(configset->query("sinTheta13_2_err").c_str()),
	::atof(configset->query("sinTheta13_2_min").c_str()),
	::atof(configset->query("sinTheta13_2_max").c_str()));
    sinTheta_2s.push_back(sinTheta13_2);
    Variable *sinTheta23_2 = configset->createVar("sinTheta23_2",
	::atof(configset->query("sinTheta23_2_init").c_str()),
	::atof(configset->query("sinTheta23_2_err").c_str()),
	::atof(configset->query("sinTheta23_2_min").c_str()),
	::atof(configset->query("sinTheta23_2_max").c_str()));
    sinTheta_2s.push_back(sinTheta23_2);
    dataset->set("sinThetas", sinTheta_2s);
    std::vector<Variable*> deltaM2s;
    Variable *deltaM221 = configset->createVar("deltaM221",
	::atof(configset->query("deltaM221_init").c_str()),
	::atof(configset->query("deltaM221_err").c_str()),
	::atof(configset->query("deltaM221_min").c_str()),
	::atof(configset->query("deltaM221_max").c_str()));
    deltaM2s.push_back(deltaM221);
    Variable *deltaM231 = configset->createVar("deltaM231",
	::atof(configset->query("deltaM231_init").c_str()),
	::atof(configset->query("deltaM231_err").c_str()),
	::atof(configset->query("deltaM231_min").c_str()),
	::atof(configset->query("deltaM231_max").c_str()));
    deltaM2s.push_back(deltaM231);
    dataset->set("deltaM2s", deltaM2s);
  } catch (GooStatsException &ex) {
    std::cout<<"Exception caught during fetching parameter configurations. probably you missed iterms in your configuration files. Read the READ me to find more details"<<std::endl;
    std::cout<<"If you think this is a bug, please email to Xuefeng Ding<xuefeng.ding.physics@gmail.com> or open an issue on github"<<std::endl;
    throw ex;
  }
  return true; 
}
