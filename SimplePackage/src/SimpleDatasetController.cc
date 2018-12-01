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
    dataset->set("exposure", ::atof(configset->query("exposure").c_str()));
    Variable *Evis = configset->createVar(configset->query("EvisVariable"),0,0,
	::atof(configset->query("Evis_min").c_str()),
	::atof(configset->query("Evis_max").c_str()));
    Evis->numbins = ::atoi(configset->query("Evis_nbins").c_str());
    dataset->set("Evis", Evis);

    if(configset->has("anaScaling")) {
      Variable *EvisFine = configset->createVar(configset->query("EvisVariable")+"_fine",0,0,
	  Evis->lowerlimit,Evis->upperlimit);
      int scale = ::atoi(configset->query("anaScaling").c_str());
      EvisFine->numbins = Evis->numbins*scale;
      dataset->set("EvisFine", EvisFine);
      dataset->set("anaScaling", scale);
    } 

    std::vector<std::string> components(GooStats::Utility::splitter(configset->query("components"),":"));;
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
      std::string type = configset->query(component+"_type");
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
		::atof(configset->query(component+"_Evis_init").c_str()),
		::atof(configset->query(component+"_Evis_err").c_str()),
		::atof(configset->query(component+"_Evis_min").c_str()),
		::atof(configset->query(component+"_Evis_max").c_str())));
	} else {
	  useNL = true;
	  Variable *inner_E = configset->createVar(component+"_inner_E",0,0,
	      ::atof(configset->query(component+"_inner_min").c_str()),
	      ::atof(configset->query(component+"_inner_max").c_str()));
	  inner_E->numbins = ::atof(configset->query(component+"_inner_nbins").c_str());
	  dataset->set(component+"_inner_E", inner_E); // energy

	  std::string inner_type = configset->query(component+"_inner_type");
	  dataset->set(component+"_inner_type",inner_type);
	  if(inner_type=="MC") {
	    dataset->set(component+"_inner_freeMCscale",
		configset->hasAndYes(component+"_inner_freeMCscale"));
	    dataset->set(component+"_inner_freeMCshift",
		configset->hasAndYes(component+"_inner_freeMCshift"));
	  } else if(type=="AnaShifted") {
	    dataset->set(component+"_dEvis", 
		configset->createVar(component+"_dEvis",
		  ::atof(configset->query(component+"_dEvis_init").c_str()),
		  ::atof(configset->query(component+"_dEvis_err").c_str()),
		  ::atof(configset->query(component+"_dEvis_min").c_str()),
		  ::atof(configset->query(component+"_dEvis_max").c_str())));
	  }
	}
      } 
    }
    if(useAna) {
      dataset->set("RPFtype",configset->query("RPFtype"));
      dataset->set("NLtype",configset->query("NLtype"));
      if(configset->has("feq"))
	dataset->set("feq",::atof(configset->query("feq").c_str()));
      else
	dataset->set("feq",1.0);
      if(useNL) {
	std::vector<Variable*> NL;
	Variable *LY = configset->createVar("LY",
	    ::atof(configset->query("LY_init").c_str()),
	    ::atof(configset->query("LY_err").c_str()),
	    ::atof(configset->query("LY_min").c_str()),
	    ::atof(configset->query("LY_max").c_str()));
	NL.push_back(LY);
	if(configset->has("NLtype") && configset->query("NLtype")=="expPar") {
	  Variable *NL_b = configset->createVar("NL_b",
	      ::atof(configset->query("NL_b_init").c_str()),
	      ::atof(configset->query("NL_b_err").c_str()),
	      ::atof(configset->query("NL_b_min").c_str()),
	      ::atof(configset->query("NL_b_max").c_str()));
	  NL.push_back(NL_b);
	  Variable *NL_c = configset->createVar("NL_c",
	      ::atof(configset->query("NL_c_init").c_str()),
	      ::atof(configset->query("NL_c_err").c_str()),
	      ::atof(configset->query("NL_c_min").c_str()),
	      ::atof(configset->query("NL_c_max").c_str()));
	  NL.push_back(NL_c);
	  Variable *NL_e = configset->createVar("NL_e",
	      ::atof(configset->query("NL_e_init").c_str()),
	      ::atof(configset->query("NL_e_err").c_str()),
	      ::atof(configset->query("NL_e_min").c_str()),
	      ::atof(configset->query("NL_e_max").c_str()));
	  NL.push_back(NL_e);
	  Variable *NL_f = configset->createVar("NL_f",
	      ::atof(configset->query("NL_f_init").c_str()),
	      ::atof(configset->query("NL_f_err").c_str()),
	      ::atof(configset->query("NL_f_min").c_str()),
	      ::atof(configset->query("NL_f_max").c_str()));
	  NL.push_back(NL_f);
	} else {
	  Variable *qc1 = configset->createVar("qc1",
	      ::atof(configset->query("qc1_init").c_str()),
	      ::atof(configset->query("qc1_err").c_str()),
	      ::atof(configset->query("qc1_min").c_str()),
	      ::atof(configset->query("qc1_max").c_str()));
	  NL.push_back(qc1);
	  Variable *qc2 = configset->createVar("qc2",
	      ::atof(configset->query("qc2_init").c_str()),
	      ::atof(configset->query("qc2_err").c_str()),
	      ::atof(configset->query("qc2_min").c_str()),
	      ::atof(configset->query("qc2_max").c_str()));
	  NL.push_back(qc2);
	}
	dataset->set("NL",NL);
      }
      std::vector<Variable*> res;
      Variable *sdn = configset->createVar("sdn",
	  ::atof(configset->query("sdn_init").c_str()),
	  ::atof(configset->query("sdn_err").c_str()),
	  ::atof(configset->query("sdn_min").c_str()),
	  ::atof(configset->query("sdn_max").c_str()));
      res.push_back(sdn);
      Variable *v1 = configset->createVar("v1",
	  ::atof(configset->query("v1_init").c_str()),
	  ::atof(configset->query("v1_err").c_str()),
	  ::atof(configset->query("v1_min").c_str()),
	  ::atof(configset->query("v1_max").c_str()));
      res.push_back(v1);
      Variable *sigmaT = configset->createVar("sigmaT",
	  ::atof(configset->query("sigmaT_init").c_str()),
	  ::atof(configset->query("sigmaT_err").c_str()),
	  ::atof(configset->query("sigmaT_min").c_str()),
	  ::atof(configset->query("sigmaT_max").c_str()));
      res.push_back(sigmaT);
      if(configset->query("RPFtype")=="MG") {
	Variable *g2 = configset->createVar("g2",
	    ::atof(configset->query("g2_init").c_str()),
	    ::atof(configset->query("g2_err").c_str()),
	    ::atof(configset->query("g2_min").c_str()),
	    ::atof(configset->query("g2_max").c_str()));
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
