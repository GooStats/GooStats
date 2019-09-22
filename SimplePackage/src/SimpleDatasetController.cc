/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "SimpleDatasetController.h"
#include "ConfigsetManager.h"
#include "GooStatsException.h"
#include "Utility.h"
bool SimpleDatasetController::collectInputs(DatasetManager *dataset) {
  try {
    dataset->set("exposure", configset->get<double>("exposure"));
    Variable *Evis = configset->createVar(configset->get("EvisVariable"),0,0,
        configset->get<double>("Evis_min"),
        configset->get<double>("Evis_max"));
    Evis->numbins = configset->get<double>("Evis_nbins");
    dataset->set("Evis", Evis);

    if(configset->has("anaScaling")) {
      Variable *EvisFine = configset->createVar(configset->get("EvisVariable")+"_fine",0,0,
          Evis->lowerlimit,Evis->upperlimit);
      int scale = configset->get<double>("anaScaling");
      EvisFine->numbins = Evis->numbins*scale;
      dataset->set("EvisFine", EvisFine);
      dataset->set("anaScaling", scale);
    } 

    std::vector<std::string> components(GooStats::Utility::splitter(configset->get("components"),":"));;
    dataset->set("components", components);

    std::vector<Variable*> Ns;
    for(auto component: components) {
      // warning: no error checking
      Variable *N = configset->var(component);
      N->numbins = -1; // dirty hack: identify them as rates
      Ns.push_back(N);
    } 
    dataset->set("Ns",Ns);

    bool useAna = false;
    bool useNL = false;
    for(auto component: components) {
      std::string type = configset->get(component+"_type");
      dataset->set(component+"_type",type);
      dataset->set(component+"_E",dataset->get<Variable*>("Evis"));
      if(type=="MC") {
        dataset->set(component+"_freeMCscale",
            configset->hasAndYes(component+"_freeMCscale"));
        dataset->set(component+"_freeMCshift",
            configset->hasAndYes(component+"_freeMCshift"));
      } else if(type.substr(0,3)=="Ana") {
        useAna = true; // NL, res, feq
        if(type=="AnaPeak") {
          dataset->set(component+"_Epeak", 
              configset->createVar(component+"_Epeak",
                configset->get<double>(component+"_Evis_init"),
                configset->get<double>(component+"_Evis_err"),
                configset->get<double>(component+"_Evis_min"),
                configset->get<double>(component+"_Evis_max")));
        } else {
          useNL = true;
          Variable *inner_E = configset->createVar(component+"_inner_E",0,0,
              configset->get<double>(component+"_inner_min"),
              configset->get<double>(component+"_inner_max"));
          inner_E->numbins = configset->get<double>(component+"_inner_nbins");
          dataset->set(component+"_inner_E", inner_E); // energy

          std::string inner_type = configset->get(component+"_inner_type");
          dataset->set(component+"_inner_type",inner_type);
          if(inner_type=="MC") {
            dataset->set(component+"_inner_freeMCscale",
                configset->hasAndYes(component+"_inner_freeMCscale"));
            dataset->set(component+"_inner_freeMCshift",
                configset->hasAndYes(component+"_inner_freeMCshift"));
          } else if(type=="AnaShifted") {
            dataset->set(component+"_dEvis", 
                configset->createVar(component+"_dEvis",
                  configset->get<double>(component+"_dEvis_init"),
                  configset->get<double>(component+"_dEvis_err"),
                  configset->get<double>(component+"_dEvis_min"),
                  configset->get<double>(component+"_dEvis_max")));
          }
        }
      } 
    }
    if(useAna) {
      dataset->set("RPFtype",configset->get("RPFtype"));
      dataset->set("NLtype",configset->get("NLtype"));
      if(configset->has("feq"))
        dataset->set("feq",configset->get<double>("feq"));
      else
        dataset->set("feq",1.0);
      if(useNL) {
        std::vector<Variable*> NL;
        Variable *LY = configset->createVar("LY",
            configset->get<double>("LY_init"),
            configset->get<double>("LY_err"),
            configset->get<double>("LY_min"),
            configset->get<double>("LY_max"));
        NL.push_back(LY);
        if(configset->has("NLtype") && configset->get("NLtype")=="expPar") {
          Variable *NL_b = configset->createVar("NL_b",
              configset->get<double>("NL_b_init"),
              configset->get<double>("NL_b_err"),
              configset->get<double>("NL_b_min"),
              configset->get<double>("NL_b_max"));
          NL.push_back(NL_b);
          Variable *NL_c = configset->createVar("NL_c",
              configset->get<double>("NL_c_init"),
              configset->get<double>("NL_c_err"),
              configset->get<double>("NL_c_min"),
              configset->get<double>("NL_c_max"));
          NL.push_back(NL_c);
          Variable *NL_e = configset->createVar("NL_e",
              configset->get<double>("NL_e_init"),
              configset->get<double>("NL_e_err"),
              configset->get<double>("NL_e_min"),
              configset->get<double>("NL_e_max"));
          NL.push_back(NL_e);
          Variable *NL_f = configset->createVar("NL_f",
              configset->get<double>("NL_f_init"),
              configset->get<double>("NL_f_err"),
              configset->get<double>("NL_f_min"),
              configset->get<double>("NL_f_max"));
          NL.push_back(NL_f);
        } else {
          Variable *qc1 = configset->createVar("qc1",
              configset->get<double>("qc1_init"),
              configset->get<double>("qc1_err"),
              configset->get<double>("qc1_min"),
              configset->get<double>("qc1_max"));
          NL.push_back(qc1);
          Variable *qc2 = configset->createVar("qc2",
              configset->get<double>("qc2_init"),
              configset->get<double>("qc2_err"),
              configset->get<double>("qc2_min"),
              configset->get<double>("qc2_max"));
          NL.push_back(qc2);
        }
        dataset->set("NL",NL);
      }
      std::vector<Variable*> res;
      Variable *sdn = configset->createVar("sdn",
          configset->get<double>("sdn_init"),
          configset->get<double>("sdn_err"),
          configset->get<double>("sdn_min"),
          configset->get<double>("sdn_max"));
      res.push_back(sdn);
      Variable *v1 = configset->createVar("v1",
          configset->get<double>("v1_init"),
          configset->get<double>("v1_err"),
          configset->get<double>("v1_min"),
          configset->get<double>("v1_max"));
      res.push_back(v1);
      Variable *sigmaT = configset->createVar("sigmaT",
          configset->get<double>("sigmaT_init"),
          configset->get<double>("sigmaT_err"),
          configset->get<double>("sigmaT_min"),
          configset->get<double>("sigmaT_max"));
      res.push_back(sigmaT);
      if(configset->get("RPFtype")=="MG") {
        Variable *g2 = configset->createVar("g2",
            configset->get<double>("g2_init"),
            configset->get<double>("g2_err"),
            configset->get<double>("g2_min"),
            configset->get<double>("g2_max"));
        res.push_back(g2);
      }
      dataset->set("res",res);
    }
  } catch (GooStatsException &ex) {
    std::cout<<"Exception caught during fetching parameter configurations. probably you missed iterms in your configuration files. Read the READ me to find more details"<<std::endl;
    std::cout<<"If you think this is a bug, please email to Xuefeng Ding<xuefeng.ding.physics@gmail.com> or open an issue on github"<<std::endl;
    throw ex;
  }
  return true; 
}
#include "SumPdf.h"
bool SimpleDatasetController::buildLikelihoods(DatasetManager *dataset) {
  GooPdf *pdf = new SumPdf(dataset->name(),
      dataset->get<double>(std::string("exposure")),
      dataset->get<std::vector<Variable*>>(std::string("Ns")),
      dataset->get<std::vector<PdfBase*>>(std::string("pdfs")),
      dataset->get<Variable*>(std::string("Evis")));
  this->setLikelihood(dataset,pdf);
  return true; 
};
