/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "SpectrumBuilder.h"
#include "GooStatsException.h"
#include "DatasetManager.h"
#include "goofit/BinnedDataSet.h"
#include "HistogramPdf.h"
#include "ResponseFunctionPdf.h"
#include "RawSpectrumProvider.h"
#include "GeneralConvolutionPdf.h"
#include "IntegralInsideBinPdf.h"
GooPdf *SpectrumBuilder::buildSpectrum(const std::string &name,DatasetManager *dataset) {
  GooPdf *pdf = this->BasicSpectrumBuilder::buildSpectrum(name,dataset);
  if(pdf) return pdf;
  const std::string &type = dataset->get<std::string>(name+"_type");
  if(type=="MC") {
    return buildMC(name,dataset);
  } else if(type=="Ana") {
    return buildAna(name,dataset);
  } else if(type=="AnaShifted") {
    return buildAnaShifted(name,dataset);
  } else if(type=="AnaPeak") {
    return buildAnaPeak(name,dataset);
  } else if(type=="placeholder") {
    return nullptr;
  } else {
    throw GooStatsException("Cannot build spectrum ["+name+"] unknown/empty type ["+type+"]");
  }
}
GooPdf *SpectrumBuilder::buildMC(const std::string &name,
    DatasetManager *dataset) {
  std::string pdfName = dataset->name()+"."+name;
  Variable *Evis = dataset->get<Variable*>(name); // Eraw
  BinnedDataSet *binned_data = loadRawSpectrum(Evis,name);
  bool freeScale = dataset->get<bool>(name+"_freeMCscale");
  bool freeShift = dataset->get<bool>(name+"_freeMCshift");
  if(freeScale||freeShift) {
    Variable *MCscale = dataset->get<Variable*>(name+"_scale");
    Variable *MCshift = dataset->get<Variable*>(name+"_shift");
    return new HistogramPdf(pdfName, binned_data , MCscale,MCshift,true /*already normalized*/);
  } else {
    return new HistogramPdf(pdfName, binned_data, nullptr, nullptr,true /*already normalized*/);
  }
}
GooPdf *SpectrumBuilder::buildAna(const std::string &name,DatasetManager *dataset) {
  std::string pdfName = dataset->name()+"."+name;
  GooPdf *resolutionPdf = new ResponseFunctionPdf(pdfName+"_RPF",
      dataset->get<Variable*>("Evis"), // Evis
      dataset->get<Variable*>(name+"_Eraw"), // Eraw
      dataset->get<std::string>("RPFtype"), // response function type
      dataset->get<std::string>("NLtype"), // non-linearity type
      dataset->get<std::vector<Variable*>>("NL"), // non-linearity
      dataset->get<std::vector<Variable*>>("res"), // resolution
      dataset->get<double>("feq")); // peak position
  dataset->set<PdfBase*>(name+"_RPF",resolutionPdf);
  return buildAnaBasic(name,dataset);
}
GooPdf *SpectrumBuilder::buildAnaShifted(const std::string &name,DatasetManager *dataset) {
  std::string pdfName = dataset->name()+"."+name;
  GooPdf *resolutionPdf = new ResponseFunctionPdf(pdfName+"_RPF",
      dataset->get<Variable*>("Evis"), // Evis
      dataset->get<Variable*>(name+"_Eraw"), // Eraw
      dataset->get<std::string>("RPFtype"), // response function type
      dataset->get<std::string>("NLtype"), // non-linearity type
      dataset->get<std::vector<Variable*>>("NL"), // non-linearity
      dataset->get<std::vector<Variable*>>("res"), // resolution
      dataset->get<double>("feq"),
      dataset->get<Variable*>(name+"_dEvis")); // peak position
  dataset->set<PdfBase*>(name+"_RPF",resolutionPdf);
  return buildAnaBasic(name,dataset);
}
GooPdf *SpectrumBuilder::buildAnaPeak(const std::string &name,DatasetManager *dataset) {
  std::string pdfName = dataset->name()+"."+name;
  GooPdf *resolutionPdf = new ResponseFunctionPdf(pdfName,
      dataset->get<Variable*>("Evis"), // Evis
      dataset->get<Variable*>(name+"_Eraw"), // Eraw
      dataset->get<std::string>("RPFtype"), // response function type
      dataset->get<std::string>("NLtype"), // non-linearity type
      dataset->get<std::vector<Variable*>>("res"), // resolution
      dataset->get<double>("feq"),
      dataset->get<Variable*>(name+"_Evis")); // peak position
  return resolutionPdf;
}
GooPdf *SpectrumBuilder::buildTODO(const std::string &name,DatasetManager *dataset) {
  std::string pdfName = dataset->name()+"."+name;
  std::cerr<<"Species ["<<pdfName<<"] not implemented!"<<endl;
  throw GooStatsException("Species not implemented");
}
BinnedDataSet *SpectrumBuilder::loadRawSpectrum(Variable *x,const std::string &name) {
  if(!provider) {
    throw GooStatsException("Trying to load spectrum with RawSpectrumProvider while not set");
  }
  BinnedDataSet* binned_data = new BinnedDataSet(x,name+"_"+x->name);
  for(int i = 0;i<x->numbins;++i) 
    binned_data->setBinContent(i,provider->pdf(name)[i]);
  return binned_data;
}
GooPdf *SpectrumBuilder::buildAnaBasic(const std::string &name,DatasetManager *dataset) {
  std::string pdfName = dataset->name()+"."+name;
  GooPdf *anaFinePdf = new GeneralConvolutionPdf(pdfName+"_fine",
      dataset->get<Variable*>("EvisFine"),
      dataset->get<Variable*>(name+"_Eraw"),
      static_cast<GooPdf*>(dataset->get<PdfBase*>(name+"_ErawPdf")),
      static_cast<GooPdf*>(dataset->get<PdfBase*>(name+"_RPF")));
  return new IntegralInsideBinPdf(pdfName,
      dataset->get<Variable*>("Evis"),
      static_cast<unsigned int>(dataset->get<int>("anaScaling")),
      anaFinePdf);
}
