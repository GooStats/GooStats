/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "solarB8SpectrumBuilder.h"
#include "DatasetManager.h"
#include "SolarNuOscPdf.h"
#include "GooStatsException.h"
#include "IBDPdf.h"
#include "NeutrinoOscillationPdf.h"
GooPdf *solarB8SpectrumBuilder::buildSpectrum(const std::string &name,
    DatasetManager *dataset) {
  GooPdf *pdf = this->BasicSpectrumBuilder::buildSpectrum(name,dataset);
  if(pdf) return pdf;
  std::string type = dataset->get<std::string>(name+"_type");
  if(type=="solarB8") {
    return buildsolarB8(name,dataset);
  } else if(type=="OscillatedsolarB8") {
    return buildOscillatedsolarB8(name,dataset);
  } else {
    return nullptr;
  }
}
#include "ProductPdf.h"
GooPdf *solarB8SpectrumBuilder::buildsolarB8(const std::string &name,
    DatasetManager *dataset) {
  return _buildOscillatedsolarB8(name,dataset,false);
}
GooPdf *solarB8SpectrumBuilder::buildOscillatedsolarB8(const std::string &name,
    DatasetManager *dataset) {
  return _buildOscillatedsolarB8(name,dataset,true);
}
GooPdf *solarB8SpectrumBuilder::_buildOscillatedsolarB8(const std::string &name,
    DatasetManager *dataset,bool ) {
  std::string pdfName = dataset->name()+"."+name;
  Variable *E = dataset->get<Variable*>(name+"_E"); 
  PdfBase *b8 = dataset->get<PdfBase*>("B8EnuPdf");
  //GooPdf *es_nuE = new ESPdf(pdfName+"_IBD",E,ESPdf::nuE);
  //GooPdf *es_nuX = new ESPdf(pdfName+"_IBD",E,ESPdf::nuX);
  GooPdf *osc = new SolarNuOscPdf(pdfName+"_osc",E,
      dataset->get<std::vector<Variable*>>("sinThetas"),
      dataset->get<std::vector<Variable*>>("deltaM2s"),
      dataset->get<double>("Ne"),true); // Pee
  //GooPdf *osc1m = new SolarNuOscPdf(pdfName+"_osc",E,
  //    dataset->get<std::vector<Variable*>>("sinThetas"),
  //    dataset->get<std::vector<Variable*>>("deltaM2s"),
  //    dataset->get<double>("Ne"),false); // 1-Pee
  std::vector<PdfBase*> NuEcomp,NuXcomp;
  NuEcomp.push_back(b8);
  NuEcomp.push_back(osc);
  //NuEcomp.push_back(es_nuE);
  GooPdf *nuEES = new ProductPdf(name,NuEcomp,E,dataset->get<double>("NHatomPerkton"));
  //NuXcomp.push_back(b8);
  //NuXcomp.push_back(osc1m);
  ////NuXcomp.push_back(es_nuX);
  //GooPdf *nuEES = new ProductPdf(name,NuXcomp,E,dataset->get<double>("NHatomPerkton"));
  // final formula:
  // f[x] = phi(E) [ #nu per day x MeV x cm^2 ]
  // 		* sigma(E) [ cm^2 ]
  // 		* Pee [ 1 ]
  // 		* NHatmPerkm [ per kt ]
  // 	unit: #IBD per (day x kt)
  // here E is energy of the neutrino, rather than Evis
  return nuEES;
}
