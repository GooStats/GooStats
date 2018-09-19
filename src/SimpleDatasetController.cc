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
DatasetManager *SimpleDatasetController::createDataset() {
  return new DatasetManager(std::string("main")); 
}
#include "ConfigsetManager.h"
#include "GooStatsException.h"
#include "Utility.h"
bool SimpleDatasetController::collectInputs(DatasetManager *dataset) {
  try {
    dataset->set("exposure", ::atof(configset->query("exposure").c_str()));
    Variable *Evis = configset->createVar(configset->query("EvisVariable"),0,0,
	::atof(configset->query("Evis_min").c_str()),
	::atof(configset->query("Evis_max").c_str()));
    Evis->numbins = ::atof(configset->query("Evis_nbins").c_str());
    dataset->set("Evis", Evis);

    std::vector<std::string> components(GooStats::Utility::splitter(configset->query("components"),":"));;
    dataset->set("components", components);

    std::vector<Variable*> Ns;
    for(auto component: components) {
      // warning: no error checking
      Variable *N = configset->createVar("N"+component,
	  ::atof(configset->query("N"+component+"_init").c_str()),
	  ::atof(configset->query("N"+component+"_err").c_str()),
	  ::atof(configset->query("N"+component+"_min").c_str()),
	  ::atof(configset->query("N"+component+"_max").c_str()));
      N->numbins = -1; // dirty hack: identify them as rates
      Ns.push_back(N);
    } 
    dataset->set("Ns",Ns);

    bool useAna = false;
    for(auto component: components) {
      std::string type = configset->query(component+"_type");
      dataset->set(component+"_type",type);
      dataset->set(component,dataset->get<Variable*>("Evis"));
      if(type=="MC") {
	dataset->set(component+"_freeMCscale",
	    configset->hasAndYes(component+"_freeMCscale"));
	dataset->set(component+"_freeMCshift",
	    configset->hasAndYes(component+"_freeMCshift"));
      } else if(type.substr(0,3)=="Ana") {
	useAna = true; // NL, res, feq

	std::string Eraw_type = configset->query(component+"_Eraw_type");
	dataset->set(component+"_Eraw_type",Eraw_type);
	if(Eraw_type=="MC") {
	  dataset->set(component+"_Eraw_freeMCscale",
	      configset->hasAndYes(component+"_Eraw_freeMCscale"));
	  dataset->set(component+"_Eraw_freeMCshift",
	      configset->hasAndYes(component+"_Eraw_freeMCshift"));
	}
	Variable *Eraw = configset->createVar(component+"_Eraw",0,0,
	    ::atof(configset->query(component+"_Eraw_min").c_str()),
	    ::atof(configset->query(component+"_Eraw_max").c_str()));
	Eraw->numbins = ::atof(configset->query(component+"_Eraw_nbins").c_str());
	dataset->set(component+"_Eraw", Eraw); // energy
	if(type=="AnaShifted") {
	  dataset->set(component+"_dEvis", 
	      configset->createVar(component+"_dEvis",
		::atof(configset->query(component+"_dEvis_init").c_str()),
		::atof(configset->query(component+"_dEvis_err").c_str()),
		::atof(configset->query(component+"_dEvis_min").c_str()),
		::atof(configset->query(component+"_dEvis_max").c_str())));
	} else if(type=="AnaPeak") {
	  dataset->set(component+"_Evis", 
	      configset->createVar(component+"_Evis",
		::atof(configset->query(component+"_Evis_init").c_str()),
		::atof(configset->query(component+"_Evis_err").c_str()),
		::atof(configset->query(component+"_Evis_min").c_str()),
		::atof(configset->query(component+"_Evis_max").c_str())));
	}
      } 
    }
    if(useAna) {
      if(configset->has("feq"))
	dataset->set("feq",::atof(configset->query("feq").c_str()));
      else
	dataset->set("feq",1.0);
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
	dataset->set("NLtype",std::string("expPar"));
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
	dataset->set("NLtype",std::string("Mach4"));
      }
      dataset->set("NL",NL);
      std::vector<Variable*> res;
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
      dataset->set("res",res);
    }
  } catch (GooStatsException &ex) {
    std::cout<<"Exception caught during fetching parameter configurations. probably you missed iterms in your configuration files. Read the READ me to find more details"<<std::endl;
    std::cout<<"If you think this is a bug, please email to Xuefeng Ding<xuefeng.ding.physics@gmail.com> or open an issue on github"<<std::endl;
    throw ex;
  }
  return true; 
}
bool SimpleDatasetController::configureParameters(DatasetManager *) {
  return true; 
};
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
