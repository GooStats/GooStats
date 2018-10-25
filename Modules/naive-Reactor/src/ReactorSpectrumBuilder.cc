/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ReactorSpectrumBuilder.h"
#include "DatasetManager.h"
#include "ReactorSpectrumPdf.h"
#include "GooStatsException.h"
#include "IBDPdf.h"
#include "NeutrinoOscillationPdf.h"
GooPdf *ReactorSpectrumBuilder::buildSpectrum(const std::string &name,
    DatasetManager *dataset) {
  GooPdf *pdf = this->BasicSpectrumBuilder::buildSpectrum(name,dataset);
  if(pdf) return pdf;
  std::string type = dataset->get<std::string>(name+"_type");
  if(type=="Reactor") {
    return buildReactor(name,dataset);
  } else if(type=="OscillatedReactor") {
    return buildOscillatedReactor(name,dataset);
  } else {
    return nullptr;
  }
}
#include "ProductPdf.h"
GooPdf *ReactorSpectrumBuilder::buildReactor(const std::string &name,
    DatasetManager *dataset) {
  return _buildOscillatedReactor(name,dataset,false);
}
GooPdf *ReactorSpectrumBuilder::buildOscillatedReactor(const std::string &name,
    DatasetManager *dataset) {
  return _buildOscillatedReactor(name,dataset,true);
}
GooPdf *ReactorSpectrumBuilder::_buildOscillatedReactor(const std::string &name,
    DatasetManager *dataset,bool oscOn) {
  std::string pdfName = dataset->name()+"."+name;
  Variable *E = dataset->get<Variable*>(name+"_E"); 
  GooPdf *reactor = new ReactorSpectrumPdf(pdfName+"_reactor",E,
      dataset->get<std::vector<Variable*>>("fractions"),
      dataset->get<std::vector<double>>("coefficients"),
      dataset->get<double>("reactorPower"),
      dataset->get<double>("distance"));
  GooPdf *ibd = new IBDPdf(pdfName+"_IBD",E);
  GooPdf *osc = new NeutrinoOscillationPdf(pdfName+"_osc",E,
      dataset->get<std::vector<Variable*>>("sinThetas"),
      dataset->get<std::vector<Variable*>>("deltaM2s"),
      dataset->get<double>("distance"));
  std::vector<PdfBase*> components;
  components.push_back(reactor);
  if(oscOn) components.push_back(osc);
  components.push_back(ibd);
  // final formula:
  // f[x] = phi(E) [ #nu per day x MeV x cm^2 ]
  // 		* sigma(E) [ cm^2 ]
  // 		* Pee [ 1 ]
  // 		* NHatmPerkm [ per kt ]
  // 	unit: #IBD per (day x kt)
  // here E is energy of the neutrino, rather than Evis
  return new ProductPdf(name,components,E,dataset->get<double>("NHatomPerkton"));
}
