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

#include "DatasetManager.h"
#include "GooStatsException.h"
#include "InputManager.h"
#include "OutputManager.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TFile.h"
#include "TH1D.h"
#include "TLegend.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TextOutputManager.h"
#include "goofit/BinnedDataSet.h"
#include "goofit/PDFs/GooPdf.h"
#include "goofit/PDFs/SumPdf.h"
bool PlotManager::init() {
  set_gStyle();
  toBeSaved.clear();
  toBeSaved.insert(gStyle);
  return true;
}
bool PlotManager::finish() {
  getOutputManager()->getOutputFile()->cd();
  for (auto obj : toBeSaved) {
    if (obj)
      obj->Write();
  }
  return true;
}
const std::string &PlotManager::outName() const { return getInputManager()->getOutputFileName(); }
void PlotManager::set_gStyle() {
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);  // it's not working when you draw from root file.
  gStyle->SetOptTitle(0);
  gStyle->SetLegendBorderSize(0);
  gStyle->SetLegendFont(132);       //Only available for ROOT version >= 5.30
  gStyle->SetLegendFillColor(0);    //Only available for ROOT version >= 5.30
  gStyle->SetLabelFont(132, "xy");  //Only available for ROOT version >= 5.30
  gStyle->SetTitleFont(132, "xy");  //Only available for ROOT version >= 5.30
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.12);
  gStyle->SetPadLeftMargin(0.13);
  gStyle->SetPadRightMargin(0.02);
  gStyle->SetTitleSize(0.04, "xy");
  gStyle->SetLabelSize(0.04, "xy");
  gStyle->SetTitleOffset(1.25, "x");
  gStyle->SetTitleOffset(1.5, "y");
  gStyle->SetLabelOffset(0.01, "xy");
  gStyle->SetOptLogx(0);
  gStyle->SetOptLogy(0);
}
void PlotManager::draw(int, const std::vector<DatasetManager *> &ds) {
  std::vector<DatasetManager *> drawable_datasets;
  for (auto d : ds) {
    if (!dynamic_cast<SumPdf *>(d->getLikelihood()))
      continue;
    drawable_datasets.push_back(d);
  }
  auto cc = drawSingleGroup(outName(), drawable_datasets);
  if (!cc)
    return;
  toBeSaved.insert(cc);
  if (createPdf())
    cc->Print((outName() + ".pdf").c_str(), "Title:merged");
  delete cc;
}
#include "GSFitManager.h"
TCanvas *PlotManager::drawSingleGroup(const std::string &name, const std::vector<DatasetManager *> &datasets) {
  int w = 1, h = 1;
  switch (datasets.size()) {
    case 0:
      return nullptr;
    case 1:
      break;
    case 2:
      w = 2;
      h = 1;
      break;
    case 4:
      w = 2;
      h = 2;
      break;
    case 6:
      w = 2;
      h = 3;
      break;
    default:
      int N = datasets.size();
      h = 1;
      w = N / h + (N % h != 0);
      while (w > 5) {
        ++h;
        w = N / h + (N % h != 0);
      }
      break;
  }
  TCanvas *cc = static_cast<TCanvas *>(gROOT->GetListOfCanvases()->FindObject((name + "_cc").c_str()));
  if (cc)
    return nullptr;
  cc = new TCanvas((name + "_cc").c_str(), ("plot for " + name).c_str(), 500 * w, 500 / 0.7 * h);
  cc->Divide(w, h);
  for (size_t i = 1; i <= datasets.size(); ++i) {
    auto dataset = datasets.at(i - 1);
    cc->cd(i);
    if (dataset->has<bool>("logy") && dataset->get<bool>("logy"))
      gPad->SetLogy(1);
    else
      gPad->SetLogy(0);
    if (dataset->has<bool>("logx") && dataset->get<bool>("logx"))
      gPad->SetLogx(1);
    else
      gPad->SetLogx(0);
    SumPdf *sumpdf = dynamic_cast<SumPdf *>(dataset->getLikelihood());
    if (!sumpdf)
      continue;
    std::map<std::string, Config> configs;
    const auto &components(dataset->get<std::vector<std::string>>("components"));
    for (auto component : components) {
      if (!dataset->has<int>(component + "_style") || !dataset->has<int>(component + "_width"))
        continue;
      const auto &style(dataset->get<int>(component + "_style"));
      const auto &width(dataset->get<int>(component + "_width"));
      int color;
      if (dataset->has<std::string>(component + "_color"))
        color = getColor(dataset->get<std::string>(component + "_color"));
      else if (dataset->has<int>(component + "_color"))
        color = dataset->get<int>(component + "_color");
      else
        continue;
      configs.insert({component, {color, style + (style == 0), width + (width == 0)}});
    }
    if (configs.size() == 0)
      std::cout << "colors/linestyles are not set or size are not consistent, default styles are used" << std::endl;
    draw(getGSFitManager(), sumpdf, configs, i, dataset);
  }

  return cc;
}
void PlotManager::draw(
    GSFitManager *gsFitManager, SumPdf *pdf, std::map<std::string, Config> config, int index, DatasetManager *dataset) {
  TPad *curr_pad = static_cast<TPad *>(gPad);
  //////////////////////////// for main pad ///////////////////////////////////
  curr_pad->cd();
  TPad *main_pad = new TPad(gPad->GetName() + TString(Form("_%d_main", index)),
                            gPad->GetTitle() + TString(Form(" %d main", index)),
                            0,
                            0.3,
                            1,
                            1);
  main_pad->cd();
  // create TLegend
  TLegend leg(0.4, 0.40, 0.96, 0.96);
  leg.SetMargin(0.15);
  leg.SetFillStyle(0);
  Variable *npe = *pdf->obsBegin();
  std::string npeFullName = npe->name;
  std::string npeName = npeFullName.substr(npeFullName.find(".") + 1);
  // -------------- data histogram
  TH1D *xvarHist = new TH1D((pdf->getName() + "_data").c_str(),
                            (pdf->getName() + "_data").c_str(),
                            npe->numbins,
                            npe->lowerlimit,
                            npe->upperlimit);
  // fill data histogram
  {
    auto data_spectra = pdf->getData();
    for (int i = 0; i < npe->numbins; ++i) {
      xvarHist->SetBinContent(i + 1, data_spectra->getBinContent(i));
      xvarHist->SetBinError(i + 1, data_spectra->getBinError(i));
    }
  }
  // set range for Y
  auto setRangeY = [](TH1 *h, bool logY) {
    std::vector<double> y;
    for (int i = 1; i <= h->GetNbinsX(); ++i)
      if (h->GetBinContent(i) > 0.1)
        y.push_back(h->GetBinContent(i));
    std::sort(y.begin(), y.end());
    //std::cout<<y.at(int(y.size()*0.01))<<" "<<y.at(int(y.size()*0.99))<<std::endl;
    double ymin = log(y.at(int(y.size() * 0.1)));
    double ymax = log(y.back());
    if (logY)
      h->GetYaxis()->SetRangeUser(exp(ymin - (ymax - ymin) * 0.1), exp(ymax + (ymax - ymin) * 0.3));
    else
      h->GetYaxis()->SetRangeUser(y.at(0) - 1 > 0 ? y.at(0) - 1 : 0, exp(ymax) + (exp(ymax) - exp(ymin)) * 0.3);
  };
  setRangeY(xvarHist, gPad->GetLogy());
  // draw data histogram
  xvarHist->GetXaxis()->SetNoExponent();
  xvarHist->GetXaxis()->SetTitle(npeName.c_str());
  xvarHist->GetYaxis()->SetTitle(Form("Events / (%s #times %.2lf %s)",
                                      TextOutputManager::get_unit().c_str(),
                                      (npe->upperlimit - npe->lowerlimit) / npe->numbins,
                                      npeName.c_str()));
  xvarHist->GetXaxis()->CenterTitle();
  xvarHist->GetYaxis()->CenterTitle();
  xvarHist->DrawClone("e");
  toBeSaved.insert(xvarHist);
  // add to legend
  leg.AddEntry(xvarHist,
               TextOutputManager::data(pdf->getName(), pdf->Norm(), "days #times tons").c_str(),
               "lpe");  // we agree it's day ton
  // -------------- total model
  pdf->cache();
  auto total_h = createTF1(pdf, 1, index);
  pdf->restore();  // GooPdf::evaluateAtPoints is called, need to restore
  // draw total model
  //total_h->SetNpx(npe->numbins);
  total_h->SetLineColor(kRed);
  total_h->SetLineWidth(4);
  total_h->SetNpx(npe->numbins);
  total_h->DrawClone("same");
  toBeSaved.insert(total_h);
  // add legend for the chi^2 etc.
  if (gsFitManager->LLfit())
    leg.AddEntry(total_h,
                 Form("-2ln(L) %.1lf p-value %.3lf #pm %.3lf",
                      gsFitManager->minus2lnlikelihood(),
                      gsFitManager->LLp(),
                      gsFitManager->LLpErr()),
                 "l");
  else
    leg.AddEntry(
        total_h,
        Form("#chi^{2}/NDF %.1lf / %d p-value %.3lf", gsFitManager->chi2(), gsFitManager->NDF(), gsFitManager->Prob()),
        "l");
  // -------------- components
  auto components = pdf->Components();
  auto Ns = pdf->Weights();
  for (size_t i = 0; i < components.size(); ++i) {
    auto pdf = dynamic_cast<GooPdf *>(components.at(i));
    auto name = pdf->getName();
    Variable *N = Ns.at(i);
    pdf->restore();  // GooPdf::evaluateAtPoints is called, need to restore
    auto single_h = createTF1(pdf, N->value * pdf->Norm(), index);
    if (config.find(pdf->getName()) != config.end()) {
      single_h->SetLineColor(config[name].color);
      single_h->SetLineStyle(config[name].style);
      single_h->SetLineWidth(config[name].width);
    } else {
      std::cerr << "Warning: line color, line style and line width not found in configuration for "
                   "["
                << name << "], set line color by order" << std::endl;
      single_h->SetLineColor(i + 2 + (i >= 3));
      single_h->SetLineStyle(1);
      single_h->SetLineWidth(2);
    }
    single_h->SetNpx(npe->numbins);
    single_h->DrawClone("same");
    toBeSaved.insert(single_h);
    leg.AddEntry(single_h,
                 TextOutputManager::rate(pdf->getName(),
                                         N->value,
                                         N->error,
                                         N->upperlimit,
                                         N->lowerlimit,
                                         N->apply_penalty,
                                         N->penalty_mean,
                                         N->penalty_sigma)
                     .c_str(),
                 "l");
  }
  // -------------- legends
  leg.ConvertNDCtoPad();
  leg.SetY1NDC(0.96 - 0.04 * leg.GetNRows());
  if (0.96 - 0.04 * leg.GetNRows() < 0.2)
    leg.SetY1NDC(0.2);
  leg.DrawClone();
  //////////////////////////// for residual pad ////////////////////////////////
  curr_pad->cd();
  TPad *res_pad = new TPad(gPad->GetName() + TString(Form("_%d_res", index)),
                           gPad->GetTitle() + TString(Form(" %d residual", index)),
                           0,
                           0,
                           1,
                           0.3);
  res_pad->cd();
  res_pad->SetLogy(false);
  res_pad->SetGridy(true);
  res_pad->SetTopMargin(0.05);
  res_pad->SetBottomMargin(0.05);
  TH1 *res = (TH1 *)xvarHist->DrawClone("e");
  res->SetName((pdf->getName() + Form("_%d_res", index)).c_str());
  res->SetTitle((pdf->getName() + Form(" %d res", index)).c_str());
  toBeSaved.insert(res);
  res->GetYaxis()->SetTitle("D-M / #sqrt{D}");
  res->GetYaxis()->SetTitleSize(0.1);
  res->GetYaxis()->SetTitleOffset(0.6);
  res->GetYaxis()->SetLabelSize(0.1);
  res->GetYaxis()->SetNdivisions(505);
  res->GetYaxis()->SetRangeUser(-3.5, 3.5);
  res->GetXaxis()->SetLabelSize(0);
  res->GetXaxis()->SetTitleSize(0);
  for (int i = 1; i <= res->GetNbinsX(); ++i) {
    double D = xvarHist->GetBinContent(i);
    double M = total_h->Eval(res->GetBinCenter(i));
    res->SetBinContent(i, (D - M) / sqrt(D + (D == 0)));
    res->SetBinError(i, 1);
  }
  // plot two pad
  curr_pad->cd();
  main_pad->Draw();
  res_pad->Draw();
}
void PlotManager::drawLikelihoodpValue(int, double LL, const std::vector<double> &LLs) {
  double mu = 0;
  double sigma = 0;
  int N = 0;
  for (auto ll : LLs) {
    ++N;
    mu += ll;
  }
  mu /= N;
  for (auto ll : LLs) {
    sigma += (ll - mu) * (ll - mu);
  }
  sigma = sqrt(sigma / (N - 1));
  mu *= 2;
  sigma *= 2;
  TH1 *h = new TH1D("m2LnLikelihood_pdf", "p.d.f. of -2Ln(Likelihood)", 100, mu - sigma * 4, mu + sigma * 4);
  for (auto ll : LLs)
    h->Fill(ll * 2);
  TCanvas *cc = static_cast<TCanvas *>(gROOT->GetListOfCanvases()->FindObject("LL_p_value"));
  if (cc)
    return;
  cc = new TCanvas("LL_p_value", "LL_p_value", 700, 500);
  h->GetXaxis()->SetTitle("-2Ln(Likelihood)");
  h->GetYaxis()->SetTitle("Entries");
  h->DrawClone();
  gPad->Update();
  gPad->Modified();
  double y_min = h->GetMaximum();
  double y_max = h->GetMinimum();
  double x_delta = gPad->GetX2() - gPad->GetX1();
  double x_min = LL * 2 - x_delta / 100;
  double x_max = LL * 2 + x_delta / 100;
  if (x_max > gPad->GetX2()) {
    double to_sub = gPad->GetX2() - x_max;
    x_min -= to_sub;
    x_max -= to_sub;
  }
  TPave pave(x_min, y_min, x_max, y_max, 0, "br");
  pave.SetFillColor(3);
  pave.SetFillStyle(3013);
  pave.SetLineWidth(2);
  pave.DrawClone();
  toBeSaved.insert(cc);
}
TF1 *PlotManager::createTF1(GooPdf *pdf, double norm, int index) {
  helpers.emplace_back(std::unique_ptr<TF1Helper>(new TF1Helper(pdf, norm, index)));
  auto tf1 = helpers.back()->getTF1();
  return tf1;
}
PlotManager::TF1Helper::TF1Helper(GooPdf *pdf, double norm, int index) {
  static int hash = -1;
  Variable *var = *pdf->obsBegin();
  double lo = var->lowerlimit;
  double up = var->upperlimit;
  double numbins = var->numbins;
  double de = (up - lo) / numbins;
  data = new TH1D(Form("%s_h_%.5lf_%d", (pdf->getName() + "_h").c_str(), norm, ++hash), "", numbins, lo, up);
  std::vector<fptype> binValues;
  // if you see ``clamping to minimum'' warning,
  // the observables in the Pdf and DataSet are consistent
  pdf->evaluateAtPoints(var, binValues);
  for (int i = 1; i <= var->numbins; ++i) {
    data->SetBinContent(i, binValues[i - 1] * de * norm);
    //    std::cout<<pdf->getName()<<" "<<i<<" "<<binValues[i-1]<<" "<<de<<" "<<norm<<std::endl;
  }
  f = new TF1((pdf->getName() + Form("_%d", index)).c_str(), this, &TF1Helper::eval, lo, up, 0, "TF1Helper", "eval");
}
double PlotManager::TF1Helper::eval(double *xx, double *) { return data->Interpolate(xx[0]); }
