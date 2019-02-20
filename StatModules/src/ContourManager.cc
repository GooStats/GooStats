/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ContourManager.h"
#include "InputManager.h"
#include "OutputManager.h"
#include "OptionManager.h"
#include "GSFitManager.h"
#include "goofit/FitManager.h"
#include "Utility.h"
#include "TMinuit.h"
#include "TMath.h"
#include "TString.h"
#include "GooStatsException.h"
#include "TFile.h"
#include "TString.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TROOT.h"
#include "TH2D.h"
#include "goofit/FitManager.h"
#include "TextOutputManager.h"
#include "TLatex.h"
bool ContourManager::init() {
  register_vars();
  setCLs();
  return true;
}
bool ContourManager::run(int) {
  plot(CLs);
  return true;
}
bool ContourManager::finish() {
  write(getOutputManager()->getOutputFile(),GlobalOption()->hasAndYes("print_contour"));
  return true;
}

void ContourManager::plot(std::vector<double> _CLs) {
  if(_CLs.size()==0) { CLs.clear(); CLs.push_back(0.90); } else CLs = _CLs;
  gMinuit = (getGSFitManager()->getFitManager()->getMinuitObject());
  minLL = (1e99);
  getGSFitManager()->getFitManager()->getMinuitValues();
  Double_t fedm; Double_t errdef; Int_t npari; Int_t nparx; Int_t istat ;
  gMinuit->mnstat  (minLL,fedm,errdef,npari,nparx,istat);
  plot_profiles();
  plot_contours();
}

void ContourManager::plot_profiles() {
  for(auto var : profiles_vars) {
    std::string title(label(var));
    TCanvas *cc = new TCanvas(GooStats::Utility::escape(var).c_str(),title.c_str(),800,600);
    cc->SetGridx(1);
    cc->SetGridy(1);
    cc->SetLogy(false);
    TGraph *gr = LLprofile(var);
    for(int i = 0;i<gr->GetN();++i) {
      gr->GetY()[i]=2*(gr->GetY()[i]-minLL);
      //std::cout<<i<<" "<<minLL<<" "<<gr->GetX()[i]<<" "<<gr->GetY()[i]<<std::endl;
    }
    double min = 1e10,L,R;
    for(double xx = get_var(var)->value; xx>gr->GetX()[0]; xx-= (gr->GetX()[1]-gr->GetX()[0])/100) {
      if(fabs(gr->Eval(xx)-1)<min) {
        min = fabs(gr->Eval(xx)-1);
        L = xx;
      }
    }
    min = 1e10;
    for(double xx = get_var(var)->value; xx<gr->GetX()[gr->GetN()-1]; xx+= (gr->GetX()[1]-gr->GetX()[0])/100) {
      if(fabs(gr->Eval(xx)-1)<min) {
        min = fabs(gr->Eval(xx)-1);
        R = xx;
      }
    }
    gr = new TGraph(gr->GetN(),gr->GetX(),gr->GetY());
    gr->Draw("ALP");
    gr->GetXaxis()->SetTitle(title.c_str());
    gr->GetYaxis()->SetTitle("-2#Delta Log(Likelihood)");
    gr->GetXaxis()->CenterTitle();
    gr->GetYaxis()->CenterTitle();
    gr->SetMarkerStyle(20);
    gr->SetMinimum(0);
    double xx[2] = { L, R };
    double yy[2] = { 1, 1 };
    TGraph *oneSigma = new TGraph(2,xx,yy);
    oneSigma->Draw("L");
    oneSigma->SetLineStyle(9);
    double xxL[2] = { L, L };
    double yyLR[2] = { 0, 1 };
    TGraph *oneSigmaL = new TGraph(2,xxL,yyLR);
    double xxR[2] = { R, R };
    TGraph *oneSigmaR = new TGraph(2,xxR,yyLR);
    oneSigmaL->Draw("L");
    oneSigmaL->SetLineStyle(9);
    oneSigmaR->Draw("L");
    oneSigmaR->SetLineStyle(9);
    int ef = TextOutputManager::get_effective_digits(R-L);
    std::string legend(label(var)+" "+TextOutputManager::show_numbers(get_var(var)->value,ef)+" ["+
      TextOutputManager::show_numbers(L-get_var(var)->value,ef)+", +"+TextOutputManager::show_numbers(R-get_var(var)->value,ef)+"]");
    TLatex *la = new TLatex;
    la->SetTextFont(132);
    la->SetNDC();
    la->SetTextSize(0.06);
    la->DrawLatex(0.26,0.89,legend.c_str());
    canvases.push_back(cc);
  }
}

void ContourManager::plot_contours() {
  for(auto var : contours_vars ) {
    std::string name(var.first+"_"+var.second);
    std::string title(label(var.first)+" vs "+label(var.second));
    TGraph *contour = dynamic_cast<TGraph*>(LLcontour(var.first,var.second,CLs));
    if(!contour) continue;
    std::cout<<"plotting contours ["<<name<<"] ["<<title<<"]"<<std::endl;
    TCanvas *cc = new TCanvas(GooStats::Utility::escape(name).c_str(),title.c_str(),800,600);
    cc->SetGridx(0);
    cc->SetGridy(0);
    cc->SetLogy(false);
    contour->SetTitle("90% CL");
    contour->GetXaxis()->SetTitle(label(var.first).c_str());
    contour->GetYaxis()->SetTitle(label(var.second).c_str());
//    TH2 *contour = dynamic_cast<TH2*>(LLcontour(var.first,var.second,CLs));
#if ROOT_VERSION_CODE > ROOT_VERSION(6,0,0)
    // new code
    contour->SetFillColorAlpha(41,0.6);
#else
    contour->SetFillColor(41);
#endif
    contour->SetFillStyle(1001);
    contour->GetXaxis()->SetLimits(
	contour->GetXaxis()->GetXmin()-(contour->GetXaxis()->GetXmax()-contour->GetXaxis()->GetXmin())*0.2,
	contour->GetXaxis()->GetXmax()+(contour->GetXaxis()->GetXmax()-contour->GetXaxis()->GetXmin())*0.2);
    contour->GetYaxis()->SetRangeUser(
	contour->GetYaxis()->GetXmin()-(contour->GetYaxis()->GetXmax()-contour->GetYaxis()->GetXmin())*0.2,
	contour->GetYaxis()->GetXmax()+(contour->GetYaxis()->GetXmax()-contour->GetYaxis()->GetXmin())*0.5);
    contour->DrawClone("ALF");
    gPad->BuildLegend(0.6566416,0.7443478,0.8571429,0.9373913);
    canvases.push_back(cc);
  }
}

void ContourManager::write(TFile *file,bool writeToPdf) {
  TDirectory *dir = gDirectory;
  file->cd();
  for( auto cc : canvases ) {
    TString name(cc->GetName());
    if(writeToPdf) cc->Print(name+".pdf");
    cc->Write();
  }
  dir->cd();
}

TGraph *ContourManager::LLprofile(const std::string &parName) {
  double left,right;
  get_par_range(parName,left,right);
  int Npoint = 40;
  if(GlobalOption()->has("profile_N")) Npoint = atoi(GlobalOption()->query("profile_N").c_str());
  int id = get_id(parName);
  std::vector<double> x,y;
  gMinuit->FixParameter(id-1);
  gMinuit->SetPrintLevel(-1);
  int fI = gMinuit->fIstrat; // the strategy
  for(int i = 0;i<Npoint;++i) {
    //if(i==1) gMinuit->Command("SET STRategy 0");
    gMinuit->Command(Form("SET PARameter %d %lf",id,left+(right-left)/(Npoint-1)*i));
    gMinuit->Migrad();
    //gMinuit->mnmigr(); // just minimize, doesn't calculate errors
    Double_t   fmin;
    Double_t   fedm;
    Double_t   errdef;
    Int_t    npari;
    Int_t    nparx;
    Int_t    istat;
    gMinuit->mnstat  (fmin,fedm,errdef,npari,nparx,istat);
    x.push_back(left+(right-left)/(Npoint-1)*i);
    y.push_back(fmin);
    printf("Profiling [%10s] (%3d/%3d;%5.2lf,%5.2lf,%5.2lf) -> %10.2lf\n",parName.c_str(),i,Npoint,x.back(),left,right,y.back());
  }
  gMinuit->Command(Form("SET STRategy %d",fI));
  gMinuit->SetPrintLevel(0);
  gMinuit->Release(id-1);
//  gMinuit->Command(Form("SCAn %d %d %lf %lf",id,Npoint,left,right));
//  TObject *obj = gMinuit->GetPlot();
//  TGraph *gr = dynamic_cast<TGraph*>(obj);
//  if(obj->ClassName()!=std::string("TGraph") || !gr) throw GooStatsException("Cannot get profile");
  TGraph *gr = new TGraph(Npoint,&x[0],&y[0]);
  return gr;
}

TObject *ContourManager::LLcontour(const std::string &par1,const std::string &par2,const std::vector<double> &) {
  int Npoint = 16;
  if(GlobalOption()->has("contour_N")) Npoint = atoi(GlobalOption()->query("contour_N").c_str());
  TGraph *contour = dynamic_cast<TGraph*>(gMinuit->Contour(Npoint,get_id(par1)-1,get_id(par2)-1));
  //  TH2D *contour = new TH2D("","",Npoint,l1,r1,Npoint,l2,r2);
  //  var1->fixed = var2->fixed = true;
  //  for(int i = 1;i<=Npoint;++i) {
  //    for(int j = 1;j<=Npoint;++j) {
  //      var1->value = contour->GetXaxis()->GetBinCenter(i);
  //      var2->value = contour->GetYaxis()->GetBinCenter(i);
  //      getGSFitManager()->getFitManager()->setupMinuit();
  //      gMinuit->mnmigr();
  //      getGSFitManager()->getFitManager()->getMinuitValues();
  //      Double_t   fmin;
  //      Double_t   fedm;
  //      Double_t   errdef;
  //      Int_t    npari;
  //      Int_t    nparx;
  //      Int_t    istat;
  //      gMinuit->mnstat  (fmin,fedm,errdef,npari,nparx,istat);
  //      contour->SetBinContent(i,j,fmin-minLL);
  //    }
  //  }
  //  contour->SetContour(CLs.size(),&CLs[0]);
  return contour;
}

void ContourManager::setCLs() {
}
Variable *ContourManager::get_var(const std::string &parName) {
  std::vector<Variable *> &vars(FitManager::vars);
  for(auto var : vars) {
    if(var->name == parName) return var;
  }
  throw GooStatsException("["+parName+"] not found");
}

int ContourManager::get_id(const std::string &parName) const {
  return getGSFitManager()->get_id(parName);
}

std::string ContourManager::label(const std::string &parName) {
  if(GlobalOption()->has("label_"+parName)) return GlobalOption()->query("label_"+parName);
  return parName;
}

void ContourManager::get_par_range(const std::string &parName,double &left,double &right) {
  Variable *var = get_var(parName);
  double bestfit = var->value;
  double sigma = var->error;
  left = bestfit-2*sigma;
  right = bestfit+2*sigma;
  auto min = [](double x,double y) { return x<y?x:y; };
  if(left<var->lowerlimit) { left = var->lowerlimit; right+=min(var->upperlimit-right,0.5*sigma); }
  if(right>var->upperlimit) { right = var->upperlimit; left-=min(left-var->lowerlimit,0.5*sigma); }
  if(GlobalOption()->has("profile_"+parName+"_min")) left = std::stod(GlobalOption()->query("profile_"+parName+"_min"));
  if(GlobalOption()->has("profile_"+parName+"_max")) right = std::stod(GlobalOption()->query("profile_"+parName+"_max"));
}

void ContourManager::register_vars() {
  if(GlobalOption()->has("plot_profiles")) {
    profiles_vars = GooStats::Utility::splitter(GlobalOption()->query("plot_profiles"),":");
    for(auto var : profiles_vars)
      std::cout<<"plot profile ["<<var<<"]"<<std::endl;
  }
  if(GlobalOption()->has("plot_contours"))
    for(auto var : GooStats::Utility::splitter(GlobalOption()->query("plot_contours"),";")) {
      auto var_pairs = GooStats::Utility::splitter(var,":");
      contours_vars.push_back(std::make_pair(var_pairs.at(0),var_pairs.at(1)));
      std::cout<<"plot contour ["<<var_pairs.at(0)<<"-"<<var_pairs.at(1)<<"]"<<std::endl;
    }
}
