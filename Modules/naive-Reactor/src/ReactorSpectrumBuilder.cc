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
GooPdf *ReactorSpectrumBuilder::buildSpectrum(const std::string &name,
    DatasetManager *dataset) {
  GooPdf *pdf = this->BasicSpectrumBuilder::buildSpectrum(name,dataset);
  if(pdf) return pdf;
  std::string type = dataset->get<std::string>(name+"_type");
  if(type=="Reactor") {
    return buildReactor(name,dataset);
  } else if(type=="OscillatedReactor") {
    return buildReactor(name,dataset);
  } else {
    return nullptr;
  }
}
GooPdf *ReactorSpectrumBuilder::buildReactor(const std::string &name,
    DatasetManager *dataset) {
  std::string pdfName = dataset->name()+"."+name;
  Variable *E = dataset->get<Variable*>(name); 
  return new ReactorSpectrumPdf(name,E,
      dataset->get<std::vector<Variable*>>("fractions"),
      dataset->get<std::vector<double>>("coefficients"));
}
#include "goofit/PDFs/ExpPdf.hh"
#include "NewExpPdf.hh"
#include "ProductPdf.h"
GooPdf *ReactorSpectrumBuilder::buildOscillatedReactor(const std::string &name,
    DatasetManager *dataset) {
  std::string pdfName = dataset->name()+"."+name;
  Variable *Evis = dataset->get<Variable*>(name); 
  GooPdf *reactor = buildReactor(name+"_noOsc",dataset);
  Variable* alpha = new Variable("alpha", -0.2, 0.1, -10, 10);
  ExpPdf* exp1 = new ExpPdf(pdfName+"_exppdf", Evis, alpha); 
  Variable* beta = new Variable("beta", -0.025, 0.1, -10, 10);
  NewExpPdf* exp2 = new NewExpPdf(pdfName+"_newExppdf", Evis, beta); 
  std::vector<PdfBase*> components;
  components.push_back(reactor);
  components.push_back(exp1);
  components.push_back(exp2);
  return new ProductPdf(name,components,Evis);
}
