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
  write(getOutputManager()->getOutputFile(),getGlobalOption()->hasAndYes("print_contour"));
  return true;
}
const OptionManager *ContourManager::getGlobalOption() const {
  return static_cast<const InputManager*>(find("InputManager"))->GlobalOption();
}
FitManager *ContourManager::getFitManager() {
  return static_cast<GSFitManager*>(find("GSFitManager"))->getFitManager();
}
OutputManager *ContourManager::getOutputManager() {
  return static_cast<OutputManager*>(find("OutputManager"));
}

void ContourManager::plot(std::vector<double> _CLs) {
  if(_CLs.size()==0) { CLs.clear(); CLs.push_back(0.90); } else CLs = _CLs;
  gMinuit = (getFitManager()->getMinuitObject());
  minLL = (1e99);
  getFitManager()->getMinuitValues();
  Double_t fedm; Double_t errdef; Int_t npari; Int_t nparx; Int_t istat ;
  gMinuit->mnstat  (minLL,fedm,errdef,npari,nparx,istat);
  plot_profiles();
  plot_contours();
}

void ContourManager::plot_profiles() {
  for(auto var : profiles_vars) {
    std::string title(label(var));
    TCanvas *cc = new TCanvas(GooStats::Utility::escape(var).c_str(),title.c_str(),800,600);
    cc->SetLogy(false);
    TGraph *gr = LLprofile(var);
    for(int i = 0;i<gr->GetN();++i) {
      gr->GetY()[i]=2*(gr->GetY()[i]-minLL);
      //std::cout<<i<<" "<<minLL<<" "<<gr->GetX()[i]<<" "<<gr->GetY()[i]<<std::endl;
    }
    gr = new TGraph(gr->GetN(),gr->GetX(),gr->GetY());
    gr->Draw("AL");
    gr->GetXaxis()->SetTitle(title.c_str());
    gr->GetYaxis()->SetTitle("#Delta Log(Likelihood)");
    gr->SetMinimum(0);
    canvases.push_back(cc);
  }
}

void ContourManager::plot_contours() {
  for(auto var : contours_vars ) {
    std::string name(var.first+"_"+var.second);
    std::string title(label(var.first)+" vs "+label(var.second));
    std::cout<<"plotting contours ["<<name<<"] ["<<title<<"]"<<std::endl;
    TCanvas *cc = new TCanvas(GooStats::Utility::escape(name).c_str(),title.c_str(),800,600);
    cc->SetLogy(false);
//    TH2 *contour = dynamic_cast<TH2*>(LLcontour(var.first,var.second,CLs));
    TGraph *contour = dynamic_cast<TGraph*>(LLcontour(var.first,var.second,CLs));
    contour->GetXaxis()->SetTitle(label(var.first).c_str());
    contour->GetYaxis()->SetTitle(label(var.second).c_str());
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
  int id = get_id(parName);
  //for(int i = 0;i<Npoint;++i) {
  //  double fix = 
  //}
  gMinuit->Command(Form("SCAn %d %d %lf %lf",id,Npoint,left,right));
  TObject *obj = gMinuit->GetPlot();
  TGraph *gr = dynamic_cast<TGraph*>(obj);
  if(obj->ClassName()!=std::string("TGraph") || !gr) throw GooStatsException("Cannot get profile");
  return gr;
}

TObject *ContourManager::LLcontour(const std::string &par1,const std::string &par2,const std::vector<double> &) {
  //  int Npoint = 30;
  //  double l1,r1,l2,r2;
  //  get_par_range(par1,l1,r1);
  //  get_par_range(par2,l2,r2);
  //  Variable *var1 = get_var(par1);
  //  Variable *var2 = get_var(par2);
  int Npoint = 16;
  if(getGlobalOption()->has("contour_N")) Npoint = atoi(getGlobalOption()->query("contour_N").c_str());
  TGraph *contour = dynamic_cast<TGraph*>(gMinuit->Contour(Npoint,get_id(par1)-1,get_id(par2)-1));
  contour->SetTitle("90% CL");
  //  TH2D *contour = new TH2D("","",Npoint,l1,r1,Npoint,l2,r2);
  //  var1->fixed = var2->fixed = true;
  //  for(int i = 1;i<=Npoint;++i) {
  //    for(int j = 1;j<=Npoint;++j) {
  //      var1->value = contour->GetXaxis()->GetBinCenter(i);
  //      var2->value = contour->GetYaxis()->GetBinCenter(i);
  //      getFitManager()->setupMinuit();
  //      gMinuit->mnmigr();
  //      getFitManager()->getMinuitValues();
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

int ContourManager::get_id(const std::string &parName) {
  unsigned int counter = 0; 
  std::vector<Variable *> &vars(FitManager::vars);
  for(auto var : vars) {
    if(var->name == parName) break;
    counter++;
  }
  if (counter == vars.size()) {
    counter = 0;
    for (std::vector<Variable*>::iterator i = vars.begin(); i != vars.end(); ++i) {
      std::cout<<counter<<" ["<<((*i)->name)<<"]"<<std::endl;
      counter++;
    }
    std::cout<<"["<<parName<<"] not found."<<std::endl;
    throw GooStatsException("par not found");
  }
  return counter+1;
}

std::string ContourManager::label(const std::string &parName) {
  if(getGlobalOption()->has("label_"+parName)) return getGlobalOption()->query("label_"+parName);
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
}

void ContourManager::register_vars() {
  if(getGlobalOption()->has("plot_profiles")) {
    profiles_vars = GooStats::Utility::splitter(getGlobalOption()->query("plot_profiles"),":");
    for(auto var : profiles_vars)
      std::cout<<"plot profile ["<<var<<"]"<<std::endl;
  }
  if(getGlobalOption()->has("plot_contours"))
    for(auto var : GooStats::Utility::splitter(getGlobalOption()->query("plot_contours"),";")) {
      auto var_pairs = GooStats::Utility::splitter(var,":");
      contours_vars.push_back(std::make_pair(var_pairs.at(0),var_pairs.at(1)));
      std::cout<<"plot contour ["<<var_pairs.at(0)<<"-"<<var_pairs.at(1)<<"]"<<std::endl;
    }
}
