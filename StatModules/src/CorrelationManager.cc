/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "CorrelationManager.h"
#include "InputManager.h"
#include "OutputManager.h"
#include "OptionManager.h"
#include "GSFitManager.h"
#include "GooStatsException.h"
#include "TMinuit.h"
#include "goofit/FitManager.h"
#include "TMath.h"
#include "OptionManager.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "Utility.h"
bool CorrelationManager::init() { 
  register_vars(); 
  return true; 
}
bool CorrelationManager::run(int) {
  print();
  analyze();
  return true;
}
bool CorrelationManager::finish() {
  if(!h||!h2r||!h2) return true;
  getOutputManager()->getOutputFile()->cd();
  h->Write();
  h2r->Write();
  h2->Write();
  return true;
}

void CorrelationManager::register_vars() {
  if(GlobalOption()->has("corr_variables")) {
    corr_vars = GooStats::Utility::split(GlobalOption()->get("corr_variables"), ":");
    for(auto var : corr_vars)
      std::cout<<"analyzing corr. for ["<<var<<"]"<<std::endl;
  }
}
TString CorrelationManager::label(TString parName) {
  if(GlobalOption()->has(("label_"+parName).Data())) return GlobalOption()->get(("label_"+parName).Data());
  return parName;
}
void CorrelationManager::print() {
  TMinuit *gMinuit = getGSFitManager()->getFitManager()->getMinuitObject();
  // global variables
  int fNpar = gMinuit->fNpar;
  int fNpagwd = gMinuit->fNpagwd;
  double *fMATUvline = gMinuit->fMATUvline;
  double *fVhmat = gMinuit->fVhmat;
  double *fGlobcc = gMinuit->fGlobcc;
  TString *fCpnam = gMinuit->fCpnam;
  double *fU = gMinuit->fU;
  double *fWerr = gMinuit->fWerr;
  int *fNiofex = gMinuit->fNiofex;
  int fNu = gMinuit->fNu;
  
  /* Local variables */
  Int_t ndex, m, n, ncoef;
  Int_t ndi, ndj;
  TString ctemp;
 
 //                                                correlation coeffs
  if (fNpar <= 1) return;
  interested_vars.clear();
  center.clear();
  covariance.clear();
  interested_names.clear();
  delete h;
  delete h2r;
  delete h2;
  getOutputManager()->getOutputFile()->cd();
  h = new TH1D("best_fit","best fit",fNpar,0,1);
  gMinuit->mnwerr();
  //    NCOEF is number of coeff. that fit on one line, not to exceed 20
  ncoef = (fNpagwd - 19) / 6;
  ncoef = TMath::Min(ncoef,20);
  ctemp = Form("%30s ","Name");
  ctemp += TString::Format(" %6s %6s","VALUE","ERROR");
  if(corr_vars.size()) {
    Printf(" ---------------------------------------------------------------------- ");
    Printf(" PARAMETER  VALUE AND ERROR ");
    Printf("%s",(const char*)ctemp);
  }
  for(int i = 1; i<=fNu;++i) {
    int l = fNiofex[i-1];
    if(l==0) continue;
    ctemp.Form("%30s ", fCpnam[i-1].Data());
    double value = fU[i-1];
    double err = fWerr[l-1];

    ctemp += TString::Format(" %6.3f %6.3f",value,err);
    h->GetXaxis()->SetBinLabel(l,label(fCpnam[i-1]));
    h->SetBinContent(l,value);
    h->SetBinError(l,err);
    //if(std::find(interested_vars.begin(),interested_vars.end(),i)==interested_vars.end()) continue;
    if(std::find(corr_vars.begin(),corr_vars.end(),fCpnam[i-1])==corr_vars.end()) continue;
    interested_vars.push_back(i);
    interested_names.push_back(fCpnam[i-1].Data());
    Printf("%s",(const char*)ctemp);
  }
  for(auto i : interested_vars) 
    center.push_back(fU[i-1]);
  h2r = new TH2D("best_fit_2D_r","best fit 2D (correlation coefficient)",fNpar,0,1,fNpar,0,1);
  h2r->GetZaxis()->SetRangeUser(-1,1);
  ctemp = Form("%30s  GLOBAL","Name");
  for( auto id : interested_vars ) {
    ctemp += TString::Format(" %6d",id);
  }
  if(interested_vars.size()) {
    Printf(" ---------------------------------------------------------------------- ");
    Printf(" PARAMETER  CORRELATION COEFFICIENTS  ");
    Printf("%s",(const char*)ctemp);
  }
  for(int ii = 1; ii<=fNu;++ii) {
    int i = fNiofex[ii-1];
    if(i==0) continue;
    ndi = i*(i + 1) / 2;
    h2r->GetXaxis()->SetBinLabel(i,label(fCpnam[ii-1]));
    h2r->GetYaxis()->SetBinLabel(i,label(fCpnam[ii-1]));
    for( int jj = 1;jj<=fNu;++jj) {
      int j = fNiofex[jj-1];
      if(j==0) continue;
      m    = TMath::Max(i,j);
      n    = TMath::Min(i,j);
      ndex = m*(m-1) / 2 + n;
      ndj  = j*(j + 1) / 2;
      fMATUvline[j-1] = fVhmat[ndex-1] / TMath::Sqrt(TMath::Abs(fVhmat[ndi-1]*fVhmat[ndj-1]));
      h2r->SetBinContent(i,j,(fMATUvline[j-1]));
    }
    if(std::find(interested_vars.begin(),interested_vars.end(),ii)==interested_vars.end()) continue;
    ctemp.Form("%30s  %7.5f ", fCpnam[ii-1].Data(),fGlobcc[i-1]);
    for( auto jj : interested_vars ) {
      int j = fNiofex[jj-1];
      ctemp += TString::Format(" %6.3f",fMATUvline[j-1]);
    }
    Printf("%s",(const char*)ctemp);
  }
  h2 = new TH2D("best_fit_2D","best fit 2D covariance",fNpar,0,1,fNpar,0,1);
  h2->GetZaxis()->SetRangeUser(-1,1);
  ctemp = Form("%30s ","Name");
  for( auto id : interested_vars ) {
    ctemp += TString::Format(" %6d",id);
  }
  if(interested_vars.size()) {
    Printf(" ---------------------------------------------------------------------- ");
    Printf(" PARAMETER  COVARIANCE ");
    Printf("%s",(const char*)ctemp);
  }
  for(int ii = 1; ii<=fNu;++ii) {
    int i = fNiofex[ii-1];
    if(i==0) continue;
    ndi = i*(i + 1) / 2;
    h2->GetXaxis()->SetBinLabel(i,label(fCpnam[ii-1]));
    h2->GetYaxis()->SetBinLabel(i,label(fCpnam[ii-1]));
    for( int jj = 1;jj<=fNu;++jj) {
      int j = fNiofex[jj-1];
      if(j==0) continue;
      m    = TMath::Max(i,j);
      n    = TMath::Min(i,j);
      ndex = m*(m-1) / 2 + n;
      ndj  = j*(j + 1) / 2;
      //      std::cout<<fVhmat[ndex-1] / TMath::Sqrt(TMath::Abs(fVhmat[ndi-1]*fVhmat[ndj-1]))<<" / "<<i<<" "<<fNiofex[i-1]<<" "<<fWerr[fNiofex[i-1]-1]<<" / "<<j<<" "<<fNiofex[j-1]<<" "<<fWerr[fNiofex[j-1]-1]<<std::endl;
      fMATUvline[j-1] = fVhmat[ndex-1] / TMath::Sqrt(TMath::Abs(fVhmat[ndi-1]*fVhmat[ndj-1])) * (fWerr[i-1]*fWerr[j-1]);
      h2->SetBinContent(i,j,fMATUvline[j-1]);
    }
    if(std::find(interested_vars.begin(),interested_vars.end(),ii)==interested_vars.end()) continue;
    ctemp.Form("%30s ", fCpnam[ii-1].Data());
    for( auto jj : interested_vars ) {
      int j = fNiofex[jj-1];
      covariance.push_back(fMATUvline[j-1]);
      ctemp += TString::Format(" %6.3f",fMATUvline[j-1]);
    }
    Printf("%s",(const char*)ctemp);
  }
  if(interested_vars.size()) Printf(" ---------------------------------------------------------------------- ");
  h->LabelsOption("v","X");
  h2r->LabelsOption("v","X");
  h2->LabelsOption("v","X");
}
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
void CorrelationManager::analyze() {
  if(!center.size()) return;
  size_t n = center.size();
  TMatrixDSym A(n,&covariance[0]);
  const TMatrixDSymEigen eig(A);
  const TVectorD eigenVal = eig.GetEigenValues();
  for(int i = 0;i<eigenVal.GetNrows();++i)
    std::cout<<"eign "<<i<<" : "<<eigenVal(i)<<std::endl;
  TMatrixD B = eig.GetEigenVectors ();
  TMatrixD Binv = B; Binv.Invert();

  Printf(" ---------------------------------------------------------------------- ");
  for(size_t i = 0;i<n;++i) {
    std::cout<<"Formula: A["<<i<<"]=";
    double sum = 0;
    for(size_t j = 0;j<n;++j) {
      printf("%5.2lf*%s%s",Binv(i,j),label(interested_names[j]).Data(),j==n-1||Binv(i,j+1)<0?" ":"+");
      sum += Binv(i,j)*center[j];
    }
    printf("=%5.2lf±%5.2lf\n",sum,sqrt(eigenVal(i)));
  }
  Printf(" ---------------------------------------------------------------------- ");
}
