/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ReactorDatasetController.h"
#include "ConfigsetManager.h"
#include "GooStatsException.h"
bool ReactorDatasetController::collectInputs(DatasetManager *dataset) {
  bool ok = this->SimpleDatasetController::collectInputs(dataset);
  if(!ok) return false;
  try {
    std::vector<Variable*> fractions;
    Variable *U235 = configset->createVar("U235",
        configset->get<double>("U235_init"),
        configset->get<double>("U235_err"),
        configset->get<double>("U235_min"),
        configset->get<double>("U235_max"));
    fractions.push_back(U235);
    Variable *U238 = configset->createVar("U238",
        configset->get<double>("U238_init"),
        configset->get<double>("U238_err"),
        configset->get<double>("U238_min"),
        configset->get<double>("U238_max"));
    fractions.push_back(U238);
    Variable *Pu239 = configset->createVar("Pu239",
        configset->get<double>("Pu239_init"),
        configset->get<double>("Pu239_err"),
        configset->get<double>("Pu239_min"),
        configset->get<double>("Pu239_max"));
    fractions.push_back(Pu239);
    Variable *Pu241 = configset->createVar("Pu241",
        configset->get<double>("Pu241_init"),
        configset->get<double>("Pu241_err"),
        configset->get<double>("Pu241_min"),
        configset->get<double>("Pu241_max"));
    fractions.push_back(Pu241);
    dataset->set("fractions", fractions);
    std::vector<double> coefficients;
    coefficients.push_back(configset->get<double>("Huber_U235_0"));
    coefficients.push_back(configset->get<double>("Huber_U235_1"));
    coefficients.push_back(configset->get<double>("Huber_U235_2"));
    coefficients.push_back(configset->get<double>("Huber_U238_0"));
    coefficients.push_back(configset->get<double>("Huber_U238_1"));
    coefficients.push_back(configset->get<double>("Huber_U238_2"));
    coefficients.push_back(configset->get<double>("Huber_Pu239_0"));
    coefficients.push_back(configset->get<double>("Huber_Pu239_1"));
    coefficients.push_back(configset->get<double>("Huber_Pu239_2"));
    coefficients.push_back(configset->get<double>("Huber_Pu241_0"));
    coefficients.push_back(configset->get<double>("Huber_Pu241_1"));
    coefficients.push_back(configset->get<double>("Huber_Pu241_2"));
    dataset->set("coefficients", coefficients);
    dataset->set("reactorPower", configset->get<double>("reactorPower"));
    dataset->set("distance", configset->get<double>("distance"));
    dataset->set("NHatomPerkton", configset->get<double>("NHatomPerkton"));
    std::vector<Variable*> sinTheta_2s;
    Variable *sinTheta12_2 = configset->createVar("sinTheta12_2",
        configset->get<double>("sinTheta12_2_init"),
        configset->get<double>("sinTheta12_2_err"),
        configset->get<double>("sinTheta12_2_min"),
        configset->get<double>("sinTheta12_2_max"));
    sinTheta_2s.push_back(sinTheta12_2);
    Variable *sinTheta13_2 = configset->createVar("sinTheta13_2",
        configset->get<double>("sinTheta13_2_init"),
        configset->get<double>("sinTheta13_2_err"),
        configset->get<double>("sinTheta13_2_min"),
        configset->get<double>("sinTheta13_2_max"));
    sinTheta_2s.push_back(sinTheta13_2);
    Variable *sinTheta23_2 = configset->createVar("sinTheta23_2",
        configset->get<double>("sinTheta23_2_init"),
        configset->get<double>("sinTheta23_2_err"),
        configset->get<double>("sinTheta23_2_min"),
        configset->get<double>("sinTheta23_2_max"));
    sinTheta_2s.push_back(sinTheta23_2);
    dataset->set("sinThetas", sinTheta_2s);
    std::vector<Variable*> deltaM2s;
    Variable *deltaM221 = configset->createVar("deltaM221",
        configset->get<double>("deltaM221_init"),
        configset->get<double>("deltaM221_err"),
        configset->get<double>("deltaM221_min"),
        configset->get<double>("deltaM221_max"));
    deltaM2s.push_back(deltaM221);
    Variable *deltaM231 = configset->createVar("deltaM231",
        configset->get<double>("deltaM231_init"),
        configset->get<double>("deltaM231_err"),
        configset->get<double>("deltaM231_min"),
        configset->get<double>("deltaM231_max"));
    deltaM2s.push_back(deltaM231);
    dataset->set("deltaM2s", deltaM2s);
  } catch (GooStatsException &ex) {
    std::cout<<"Exception caught during fetching parameter configurations. probably you missed iterms in your configuration files. Read the READ me to find more details"<<std::endl;
    std::cout<<"If you think this is a bug, please email to Xuefeng Ding<xuefeng.ding.physics@gmail.com> or open an issue on github"<<std::endl;
    throw ex;
  }
  return true; 
}
