/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "DumperPdf.h"
#include "GeneralConvolutionPdf.h"
#include "goofit/PDFs/SumPdf.h"
#include "SumLikelihoodPdf.h"
#include "goofit/FitControl.h"
#include "goofit/BinnedDataSet.h"
void dumpVar(GooPdf *pdf) {
  std::vector<Variable*> vars;
  vars.clear();
  pdf->getParameters(vars);
  for (auto var : vars) {
    printf("%s %d name %s addr %p val %lf err %lf lo %lf up %lf fixed %s\n",
        pdf->getName().c_str(),var->index, 
        var->name.c_str(), var,
        var->value, var->error, var->lowerlimit, var->upperlimit,
        (var->fixed?"yes":"no"));
  }
}
template<>
void DumperPdf<GeneralConvolutionPdf>::dumpConvolution(Variable *var) {
  this->getValue();
  double obs_lo = host_iConsts[0];
  double obs_hi = host_iConsts[1];
  double obs_numbins = host_iConsts[2];
  assert(var->numbins==obs_numbins);
  assert(var->upperlimit==obs_hi);
  assert(var->lowerlimit==obs_lo);
  double intvar_lo = host_iConsts[3];
  double intvar_hi = host_iConsts[4];
  double intvar_numbins = host_iConsts[5];
  double intvar_step = (intvar_hi-intvar_lo)/intvar_numbins;
  double obs_val = var->value;
  double obs_step = (obs_hi-obs_lo)/obs_numbins;
  const int obs_bin = (int) FLOOR((obs_val-obs_lo)/obs_step);
  thrust::host_vector<fptype> model(*modelWorkSpace);
  thrust::host_vector<fptype> rpf(*resolWorkSpace);
  printf(" %le->%le(%le) %le->%le(%le)\n",
      intvar_lo+intvar_step*0.5,intvar_hi-intvar_step*0.5,intvar_numbins,
      obs_lo+obs_step*0.5,obs_hi-obs_step*0.5,obs_numbins);
  for(int i = 0;i<intvar_numbins;++i) {
    printf(" %le(%d)->%le %le(%d)->%le\n",
        intvar_lo+intvar_step*(i+0.5),i,model[i],
        obs_val,int(obs_bin+i*obs_numbins+0.5),rpf[obs_bin+i*obs_numbins]);
  }
}
template<>
void DumperPdf<SumPdf>::dumpPdf(BinnedDataSet *data) {
  copyParams();
  dumpVar(this);
  std::vector<Variable*> vars;
  this->getParameters(vars);
  Variable *var = *(this->obsBegin());
  std::vector<GooPdf*> pdfs;
  std::vector<std::vector<fptype> > pdf_values;

  std::vector<fptype> pdf_value;
  this->evaluateAtPoints(var,pdf_value);
  pdfs.push_back(this);
  pdf_values.push_back(pdf_value);
  for(unsigned int i = 0;i<components.size();++i) {
    GooPdf *pdf = dynamic_cast<GooPdf*>(components.at(i));
    if(!pdf) {
      std::cerr<<"warning: <"<<components.at(i)->getName()<<"> is not a GooPdf"<<std::endl;
      continue;
    }
    pdf->evaluateAtPoints(var,pdf_value);
    pdfs.push_back(pdf);
    pdf_values.push_back(pdf_value);
  }
  this->setData(data);
  double step = (var->upperlimit - var->lowerlimit) / var->numbins;
  printf("Now canning val\n");
  static double result = 0;
  for(unsigned int i = 0;i<data->getNumBins();++i) {
    var->value = var->lowerlimit + (i+0.5)*step;
    double center = var->value;
    double content = data->getBinContent(i);
    double dvar = (var->upperlimit-var->lowerlimit)/var->numbins;
    double fvalue = pdf_values.at(0).at(i);
    double expEvents = fvalue*dvar;
    double measureEvents = content;
    result += fvalue>0?(expEvents - measureEvents*EVALLOG(expEvents)+::lgamma(measureEvents+1)):0; 
    printf("log(L) %lf b %lf M %lf tot %lf\n",result,center,content,fvalue*dvar);
    for(unsigned int j = 1;j<pdfs.size();++j)
      printf(" %s %le",pdfs.at(j)->getName().c_str(),pdf_values.at(j).at(i)*dvar*norm*vars.at(j-1)->value);
    printf("\n");
  }
}
template<>
void DumperPdf<SumLikelihoodPdf>::dumpPdf(const std::vector<BinnedDataSet*> &datas) {
  copyParams();
  dumpVar(this);

  Variable *var;
  std::map<PdfBase*,std::pair<std::string,std::vector<fptype> > > pdf_map; // name, value, norm
  std::vector<std::vector<fptype> > sums_value;
  for(unsigned int i = 0;i<components.size();++i) {
    std::vector<fptype> sum_value;
    SumPdf *pdf = dynamic_cast<SumPdf*>(components.at(i));
    var = *(pdf->obsBegin());
    pdf->setData(datas.at(i));
    pdf->evaluateAtPoints(var,sum_value);
    sums_value.push_back(sum_value);
    std::vector<PdfBase*> pdfs = pdf->components;
    for(unsigned int j = 0;j<pdfs.size();++j) {
      GooPdf *single_species = dynamic_cast<GooPdf*>(pdfs.at(j));
      if(pdf_map.find(static_cast<PdfBase*>(single_species))==pdf_map.end()) {
        std::vector<fptype> pdf_value;
        single_species->evaluateAtPoints(var,pdf_value);
        pdf_map.insert(make_pair(static_cast<PdfBase*>(single_species),make_pair(single_species->getName(),pdf_value)));
      }
    }
  }
  double step = (var->upperlimit - var->lowerlimit) / var->numbins;
  double dvar = step;
  double result = 0;
  for(unsigned int i = 0;i<components.size();++i) {
    SumPdf *sum_pdf = dynamic_cast<SumPdf*>(components.at(i)); // for sum_pdf->norm
    BinnedDataSet *data = datas.at(i);
    for(unsigned int j = 0;j<data->getNumBins();++j) {
      var->value = var->lowerlimit + (j+0.5)*step;
      double center = var->value;
      double content = data->getBinContent(j);
      double fvalue = sums_value.at(i).at(j);
      double expEvents = fvalue*dvar;
      double measureEvents = content;
      result += fvalue>0?(expEvents - measureEvents*EVALLOG(expEvents)+::lgamma(measureEvents+1)):0; 
      printf("log(L) %.10le b %lf M %lf tot %.10lf\n",result,center,content,fvalue*dvar);
      std::vector<PdfBase*> pdfs = sum_pdf->components;
      for(unsigned int k = 0;k<pdfs.size();++k) {
        PdfBase *pdf = pdfs.at(k);
        std::pair<std::string,std::vector<fptype> > t = pdf_map.at(pdf);
        printf(" %s %.10le",t.first.c_str(),t.second.at(j)*dvar*sum_pdf->norm*host_params[host_indices[sum_pdf->parameters+2*k+1]]);
      }
      printf("\n");
    }
  }
}
template<>
void DumperPdf<GooPdf>::dumpIndices() {
  copyParams();
  dumpVar(this);
  std::cout<<"calling inside "<<__func__<<std::endl;
  int i = 0;
  while(i<totalParams) {
    std::cout<<"*"<<pdfName[i]<<"*=>";
    std::cout<<"<|"<<host_indices[i]<<"|->|";
    for(unsigned int j = 0;j<host_indices[i];++j)
      std::cout << host_indices[j+i+1] << ",";
    i+=host_indices[i]+1;
    std::cout<<"|,("<<host_indices[i]<<")->(";
    for(unsigned int j = 0;j<host_indices[i];++j)
      std::cout << "["<<i+j+1<<"]"<<host_indices[j+i+1] << ",";
    i+=host_indices[i]+1;
    std::cout<<")>;<=**";
    std::cout<<std::endl;
  }
  std::cout<<getName()<<" has the fit control "<<fitControl<<" : binned? "<<fitControl->binnedFit()<<" binErr? "<<fitControl->binErrors()<<std::endl;
}

