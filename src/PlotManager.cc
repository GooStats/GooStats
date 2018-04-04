/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "PlotManager.h"
#include "TextOutputManager.h"
#include "DatasetManager.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TF1.h"
#include "SumPdf.h"
#include "TMath.h"
#include "goofit/PDFs/GooPdf.h"
#include "TF1.h"
#include "TFile.h"
#include "GooStatsException.h"
bool PlotManager::init() {
  out = TFile::Open((outName()+".root").c_str(),"RECREATE");
  if(!out->IsOpen()) {
    std::cout<<"Cannot create output root file: <"<<outName()<<">"<<std::endl;
    throw GooStatsException("Cannot create output root file");
  }
  set_gStyle();
  return true;
}
bool PlotManager::run() {
  return true;
}
bool PlotManager::finish() {
  cd();
  for(auto obj : toBeSaved) {
    obj->Write();
  }
  out->Close();
  return true;
}
void PlotManager::set_gStyle() {
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);       // it's not working when you draw from root file.
  gStyle->SetOptTitle(0);
  gStyle->SetLegendBorderSize(0);
  gStyle->SetLegendFont(132); //Only available for ROOT version >= 5.30
  gStyle->SetLegendFillColor(0); //Only available for ROOT version >= 5.30
  gStyle->SetLabelFont(132,"xy"); //Only available for ROOT version >= 5.30
  gStyle->SetTitleFont(132,"xy"); //Only available for ROOT version >= 5.30
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.09);
  gStyle->SetPadLeftMargin(0.13);
  gStyle->SetPadRightMargin(0.02);
  gStyle->SetTitleSize(0.04,"x");
  gStyle->SetTitleOffset(1.15,"x");
  gStyle->SetTitleOffset(1.35,"yz");
  gStyle->SetOptLogy(1);
}
void PlotManager::draw(const std::vector<DatasetManager*> &datasets) {
  auto cc = drawSingleGroup("merged",datasets);
  if(createPdf()) cc->Print((outName()+".pdf").c_str(),"Title:merged");
}
TCanvas *PlotManager::drawSingleGroup(const std::string &name,const std::vector<DatasetManager*> &datasets) {
  int w = 1,h = 1;
  switch(datasets.size()) {
    case 1: break;
    case 2: w = 2; h = 1; break;
    case 4: w = 2; h = 2; break;
    case 6: w = 2; h = 3; break;
    default:
	    int N = datasets.size();
	    h = 1;
	    w = N/h + (N%h!=0);
	    while(w>5) {
	      ++h;
	      w = N/h + (N % h !=0);
	    }
	    break;
  }
  TCanvas *cc = new TCanvas((name+"_cc").c_str(),("plot for "+name).c_str(),700*w,700*h);
  cc->Divide(w,h);
  for(size_t i = 1;i<=datasets.size();++i) {
    auto dataset = datasets.at(i-1);
    cc->cd(i);
    SumPdf *sumpdf = dynamic_cast<SumPdf*>(dataset->getLikelihood());
    if(!sumpdf) continue;
    std::map<std::string,Config> configs;
    const auto &components(dataset->get<std::vector<std::string>>("components"));
    for(auto component : components) {
      if(!dataset->has<std::string>(component+"_color") || !dataset->has<double>(component+"_linestyle")) continue;
      const auto &color(dataset->get<std::string>(component+"_color"));
      const auto &linestyle(int(dataset->get<double>(component+"_linestyle")+0.5));
      configs.insert( std::make_pair(component, (Config) { getColor(color),linestyle + (linestyle==0) } ) );
    }
    if(configs.size()==0) 
      std::cout<<"colors/linestyles are not set or size are not consistent, default styles are used"<<std::endl;
    draw(sumpdf,configs);
  }

  return cc;
}
void PlotManager::draw(SumPdf *sumpdf,std::map<std::string,Config> config) {
  toBeSaved.clear();
  TLegend *leg = new TLegend(0.5,0.40,0.95,0.99);
  leg->SetMargin(0.15);
  leg->SetFillStyle(0);
  Variable *npe = *sumpdf->obsBegin();
  std::string npeFullName = npe->name;
  std::string npeName = npeFullName.substr(npeFullName.find(".")+1);
  BinnedDataSet *data_spectra = sumpdf->getData();
  //  for (int i = 1; i <= npe->numbins; ++i)  {
  //    std::cout<<"data: "<<i<<" "<<data_spectra->getBinContent(i)<<std::endl;
  //  }
  TH1D *xvarHist = new TH1D(sumpdf->getName().c_str(), 
			    sumpdf->getName().c_str(),
			    npe->numbins, npe->lowerlimit, npe->upperlimit);
  xvarHist->GetXaxis()->SetNoExponent();
  xvarHist->GetXaxis()->SetTitle(npeName.c_str());
  xvarHist->GetYaxis()->SetTitle(Form("Events / (day #times ktons #times %d %s)",
				      int(floor((npe->upperlimit-npe->lowerlimit)/npe->numbins)),npeName.c_str()));
  xvarHist->GetXaxis()->CenterTitle();
  xvarHist->GetYaxis()->CenterTitle();
  leg->AddEntry(xvarHist,TextOutputManager::data(sumpdf->getName(),
						 sumpdf->Norm(),"days #times tons").c_str(),"lpe");
  toBeSaved.push_back(xvarHist);
  for(int i = 0;i<npe->numbins;++i) {
    xvarHist->SetBinContent(i+1,data_spectra->getBinContent(i));
    xvarHist->SetBinError(i+1,data_spectra->getBinError(i));
  }
  xvarHist->Draw("e");
  TF1Helper *total_helper = new TF1Helper(sumpdf,1);
  TF1 *total_h = total_helper->getTF1();
  sumpdf->setData(data_spectra);
  total_h->Draw("same");
  total_h->SetLineColor(kRed);
  total_h->SetLineWidth(4);
  total_h->SetNpx(npe->numbins);
  double chi2 = sumpdf->Chi2();
  int ndf = sumpdf->NDF();
  string total_legend = 
    Form("#chi^{2}/NDF %.1lf / %d p-value %.3lf",
	 chi2,ndf,TMath::Prob(chi2,ndf));
  leg->AddEntry(total_h,total_legend.c_str(),"l");
  toBeSaved.push_back(total_h);
  auto components = sumpdf->Components();
  auto Ns = sumpdf->Weights();
  for(size_t i = 0;i<components.size();++i) {
    GooPdf *pdf = static_cast<GooPdf*>(components.at(i));
    Variable *N = Ns.at(i);
    TF1Helper *single_helper = new TF1Helper(pdf,N->value*sumpdf->Norm());
    TF1 *single_h = single_helper->getTF1();
    toBeSaved.push_back(single_h);
    leg->AddEntry(single_h,TextOutputManager::rate(pdf->getName(),N->value,N->error,N->upperlimit,N->lowerlimit,"cpd/ktons",N->apply_penalty,N->penalty_mean,N->penalty_sigma).c_str(),"l");
    single_h->SetLineWidth(2);
    if(config.find(pdf->getName())!=config.end()) {
      single_h->SetLineColor(config[pdf->getName()].color);
      single_h->SetLineStyle(config[pdf->getName()].style);
    } else {
      single_h->SetLineColor(i+2+(i>=3));
      single_h->SetLineStyle(1);
    }
    single_h->SetNpx(npe->numbins);
    single_h->Draw("same");
  }
  leg->ConvertNDCtoPad();
  leg->SetY2NDC(0.96);
  leg->SetY1NDC(0.96-0.03*leg->GetNRows());
  leg->Draw();
}
PlotManager::TF1Helper::TF1Helper(GooPdf *pdf,double norm) {
  static int hash = -1;
  Variable *var = *pdf->obsBegin();
  double lo = var->lowerlimit;
  double up = var->upperlimit;
  double numbins = var->numbins;
  double de = (up-lo)/numbins;
  data = new TH1D(Form("%s_h_%.5lf_%d",(pdf->getName()+"_h").c_str(),norm,++hash), 
		  "", numbins,lo,up);
  std::vector<fptype> binValues;
  // if you see ``clamping to minimum'' warning, 
  // the observables in the Pdf and DataSet are consistent
  pdf->evaluateAtPoints(var, binValues);  
  for (int i = 1; i <= var->numbins; ++i)  {
    data->SetBinContent(i, binValues[i-1] * de *norm);
    //    std::cout<<pdf->getName()<<" "<<i<<" "<<binValues[i-1]<<" "<<de<<" "<<norm<<std::endl;
  }
  f = new TF1(pdf->getName().c_str(),this,&TF1Helper::eval,lo,up,0,"TF1Helper","eval");
};
double PlotManager::TF1Helper::eval(double *xx, double *) {
  return data->Interpolate(xx[0]);
}
void PlotManager::cd() { out->cd(); }
