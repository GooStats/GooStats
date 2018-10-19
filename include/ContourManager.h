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
#include "Module.h"
class TGraph;
#include <string>
#include <vector>
#include <utility>
class TMinuit;
class TFile;
class Variable;
class TCanvas;
class FitManager;
class OptionManager;
class OutputManager;
class TH2;
class TObject;
class ContourManager : public Module {
  public:
    ContourManager() : Module("ContourManager") { }
    bool init() final;
    bool run(int = 0/*event*/) final;
    bool finish() final;
    bool check() const final { return has("GSFitManager")&&has("InputManager")&&has("OutputManager"); }
  private:
    const OptionManager *getGlobalOption() const;
    FitManager *getFitManager();
    OutputManager *getOutputManager();

  private:
    void register_vars();
    std::string label(const std::string &parName);
    void plot(std::vector<double> _CLs = std::vector<double>());
    void write(TFile *file,bool writeToPdf = true);
  private:
    Variable *get_var(const std::string &parName);
    int get_id(const std::string &parName);
    void get_par_range(const std::string &parName,double &left,double &right);
  private:
    TGraph *LLprofile(const std::string &parName);
    TObject *LLcontour(const std::string &par1,const std::string &par2,const std::vector<double> &CLs);
    void plot_profiles();
    void plot_contours();
    void setCLs();
  private:
    std::vector<std::string> profiles_vars;
    std::vector<std::pair<std::string,std::string>> contours_vars;
    std::vector<double> CLs;
    std::vector<TCanvas*> canvases;
    TMinuit *gMinuit;
    double minLL;
};
#endif

