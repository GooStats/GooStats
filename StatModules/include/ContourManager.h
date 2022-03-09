/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ContourManager_H
#define ContourManager_H
#include "StatModule.h"
class TGraph;
#include <string>
#include <utility>
#include <vector>
class TMinuit;
class TFile;
struct Variable;
class TCanvas;
class FitManager;
class OptionManager;
class OutputManager;
class TH2;
class TObject;
class ContourManager : public StatModule {
 public:
  ContourManager(const std::string &_name = "ContourManager") : StatModule(_name) {}
  bool init() final;
  bool run(int = 0 /*event*/) final;
  bool finish() final;

 private:
  void register_vars();
  std::string label(const std::string &parName);
  void plot(std::vector<double> _CLs = std::vector<double>());
  void write(TFile *file, bool writeToPdf = true);

 private:
  Variable *get_var(const std::string &parName);
  int get_id(const std::string &parName) const;
  void get_par_range(const std::string &parName, double &left, double &right);

 private:
  TGraph *LLprofile(const std::string &parName);
  TObject *LLcontour(const std::string &par1, const std::string &par2, const std::vector<double> &CLs);
  void plot_profiles();
  void plot_contours();
  void setCLs();

 private:
  std::vector<std::string> profiles_vars;
  std::vector<std::pair<std::string, std::string>> contours_vars;
  std::vector<double> CLs;
  std::vector<TCanvas *> canvases;
  TMinuit *gMinuit;
  double minLL;
};
#endif
