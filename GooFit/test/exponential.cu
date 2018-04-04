#include "goofit/Variable.h" 
#include "goofit/FitManager.h"
#include "goofit/BinnedDataSet.h" 
#include "goofit/PDFs/ExpPdf.hh" 
#include "SumPdf.h"
#include <iostream>
#include "TF1.h"
#include "TH1D.h"

using namespace std; 

int main (int argc, char** argv) {
  // Independent variable. 
  double max = log(1 + RAND_MAX/2);
  Variable* xvar = new Variable("xvar", 0, max);
  TF1 *f = new TF1("f","expo",0,log(1 + RAND_MAX/2));
  f->SetParameters(100,-0.5);
  TH1 *h = new TH1D("h","h",xvar->numbins,0,max);
  h->FillRandom("f",1000000); // 1000, 000 events = 1000 event/(ton*day) * 1000 ton*day
  h->Fit(f);
  
  // Data set
  BinnedDataSet* data= new BinnedDataSet(xvar);
  for(int i = 0;i<xvar->numbins;++i) {
    data->setBinContent(i,h->GetBinContent(i+1));
  }

  // pdf
  Variable* alpha = new Variable("alpha", -2, 0.1, -10, 10);
  Variable* rate = new Variable("R", 500, 1, 0, 2000); // rate
  ExpPdf* exppdf = new ExpPdf("exppdf", xvar, alpha); 

  std::vector<PdfBase*> components;
  std::vector<Variable*> rates;
  components.push_back(exppdf);
  rates.push_back(rate);
  SumPdf* sum = new SumPdf("sumpdf",1000 /*exposure, day*ton*/, rates, components, xvar);
  sum->setData(data);
  FitManager fitter(sum);
//  exppdf->setData(data);
//  FitManager fitter(exppdf);

  fitter.fit(); 

  return 0;
}
