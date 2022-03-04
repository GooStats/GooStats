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
#include "goofit/Variable.h"
void recursiveSet(const std::string comp, ConfigsetManager *configset, DatasetManager *dataset, bool &useAna,
                  bool &useNL) {
  auto type = configset->get(comp + "_type");
  dataset->set(comp + "_type", type);
  if (!configset->hasVar(comp + "_E")) {
    if (configset->has(comp + "_E_nbins")) {
      auto inner_E = configset->createVar(comp + "_E", 0, 0, configset->getOrConvert(comp + "_E_min"),
                                          configset->getOrConvert(comp + "_E_max"));
      inner_E->numbins = configset->getOrConvert(comp + "_E_nbins");
    } else {
      configset->linkVar(comp + "_E", configset->get("EvisVariable"));
    }
  }
  dataset->set(comp + "_E", configset->var(comp + "_E"));// energy
  if (type == "MC") {
    dataset->set(comp + "_freeMCscale", configset->hasAndYes(comp + "_freeMCscale"));
    dataset->set(comp + "_freeMCshift", configset->hasAndYes(comp + "_freeMCshift"));
  } else if (type.substr(0, 3) == "Ana") {
    useAna = true;// NL, res, feq
    if (type == "AnaPeak") {
      dataset->set(comp + "_Epeak",
                   configset->createVar(comp + "_Epeak", configset->getOrConvert(comp + "_Evis_init"),
                                        configset->getOrConvert(comp + "_Evis_err"),
                                        configset->getOrConvert(comp + "_Evis_min"),
                                        configset->getOrConvert(comp + "_Evis_max")));
    } else {
      useNL = true;
      if (type == "AnaShifted") {
        dataset->set(comp + "_dEvis",
                     configset->createVar(comp + "_dEvis", configset->getOrConvert(comp + "_dEvis_init"),
                                          configset->getOrConvert(comp + "_dEvis_err"),
                                          configset->getOrConvert(comp + "_dEvis_min"),
                                          configset->getOrConvert(comp + "_dEvis_max")));
      }

      auto newName = comp + "_inner";
      dataset->set(comp + ".deps", newName);
      recursiveSet(newName, configset, dataset, useAna, useNL);
    }
  }
};

bool SimpleDatasetController::collectInputs() {
  try {
    dataset->set("exposure", configset->getOrConvert("exposure"));
    Variable *Evis = configset->createVar(configset->get("EvisVariable"), 0, 0, configset->getOrConvert("Evis_min"),
                                          configset->getOrConvert("Evis_max"));
    Evis->numbins = configset->getOrConvert("Evis_nbins");
    dataset->set("Evis", Evis);

    if (configset->has("anaScaling")) {
      Variable *EvisFine =
              configset->createVar(configset->get("EvisVariable") + "_fine", 0, 0, Evis->lowerlimit, Evis->upperlimit);
      int scale = configset->getOrConvert("anaScaling");
      EvisFine->numbins = Evis->numbins * scale;
      dataset->set("EvisFine", EvisFine);
      dataset->set("anaScaling", scale);
    }

    std::vector<std::string> components(GooStats::Utility::split(configset->get("components"), ":"));
    ;
    dataset->set("components", components);

    std::vector<Variable *> Ns;
    for (const auto &component: components) {
      // warning: no error checking
      Variable *N = configset->var(component);
      N->numbins = -1;// dirty hack: identify them as rates
      Ns.push_back(N);
    }
    dataset->set("Ns", Ns);

    bool useAna = false;
    bool useNL = false;

    for (const auto &component: components) { recursiveSet(component, configset, dataset.get(), useAna, useNL); }
    if (useAna) {
      dataset->set("RPFtype", configset->get("RPFtype"));
      dataset->set("NLtype", configset->get("NLtype"));
      if (configset->has("feq")) dataset->set("feq", configset->getOrConvert("feq"));
      else
        dataset->set("feq", 1.0);
      if (useNL) {
        std::vector<Variable *> NL;
        Variable *LY = configset->createVar(
                "LY", configset->getOrConvert("LY_init"), configset->getOrConvert("LY_err"), configset->getOrConvert("LY_min"), configset->getOrConvert("LY_max"));
        NL.push_back(LY);
        if (configset->has("NLtype") && configset->get("NLtype") == "expPar") {
          Variable *NL_b = configset->createVar(
                  "NL_b", configset->getOrConvert("NL_b_init"), configset->getOrConvert("NL_b_err"),
                  configset->getOrConvert("NL_b_min"), configset->getOrConvert("NL_b_max"));
          NL.push_back(NL_b);
          Variable *NL_c = configset->createVar(
                  "NL_c", configset->getOrConvert("NL_c_init"), configset->getOrConvert("NL_c_err"),
                  configset->getOrConvert("NL_c_min"), configset->getOrConvert("NL_c_max"));
          NL.push_back(NL_c);
          Variable *NL_e = configset->createVar(
                  "NL_e", configset->getOrConvert("NL_e_init"), configset->getOrConvert("NL_e_err"),
                  configset->getOrConvert("NL_e_min"), configset->getOrConvert("NL_e_max"));
          NL.push_back(NL_e);
          Variable *NL_f = configset->createVar(
                  "NL_f", configset->getOrConvert("NL_f_init"), configset->getOrConvert("NL_f_err"),
                  configset->getOrConvert("NL_f_min"), configset->getOrConvert("NL_f_max"));
          NL.push_back(NL_f);
        } else {
          Variable *qc1 = configset->createVar(
                  "qc1", configset->getOrConvert("qc1_init"),
                                               configset->getOrConvert("qc1_err"),
                                               configset->getOrConvert("qc1_min"), configset->getOrConvert("qc1_max"));
          NL.push_back(qc1);
          Variable *qc2 = configset->createVar(
                  "qc2", configset->getOrConvert("qc2_init"),
                                               configset->getOrConvert("qc2_err"),
                                               configset->getOrConvert("qc2_min"), configset->getOrConvert("qc2_max"));
          NL.push_back(qc2);
        }
        dataset->set("NL", NL);
      }
      std::vector<Variable *> res;
      Variable *sdn = configset->createVar(
              "sdn", configset->getOrConvert("sdn_init"), configset->getOrConvert("sdn_err"), configset->getOrConvert("sdn_min"), configset->getOrConvert("sdn_max"));
      res.push_back(sdn);
      Variable *v1 = configset->createVar(
              "v1", configset->getOrConvert("v1_init"), configset->getOrConvert("v1_err"),
                                   configset->getOrConvert("v1_min"), configset->getOrConvert("v1_max"));
      res.push_back(v1);
      Variable *sigmaT = configset->createVar(
              "sigmaT", configset->getOrConvert("sigmaT_init"), configset->getOrConvert("sigmaT_err"),
              configset->getOrConvert("sigmaT_min"), configset->getOrConvert("sigmaT_max"));
      res.push_back(sigmaT);
      if (configset->get("RPFtype") == "MG") {
        Variable *g2 = configset->createVar(
                "g2", configset->getOrConvert("g2_init"), configset->getOrConvert("g2_err"), configset->getOrConvert("g2_min"), configset->getOrConvert("g2_max"));
        res.push_back(g2);
      }
      dataset->set("res", res);
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
#include "goofit/PDFs/SumPdf.h"
bool SimpleDatasetController::buildLikelihood() {
  GooPdf *pdf = new SumPdf(dataset->fullName(), dataset->get<double>(std::string("exposure")),
                           dataset->get<std::vector<Variable *>>(std::string("Ns")),
                           dataset->get<std::vector<PdfBase *>>(std::string("pdfs")),
                           dataset->get<Variable *>(std::string("Evis")));
  dataset->setLikelihood(pdf);
  return true;
}
