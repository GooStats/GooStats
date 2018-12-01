/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef CorrelationManager_H
#define CorrelationManager_H
#include "StatModule.h"
class FitManager;
#include "TString.h"
class TH1;
class TH2;
class CorrelationManager : public StatModule {
  public:
    CorrelationManager(const std::string &_name="CorrelationManager") : StatModule(_name) { }
    bool init() final;
    bool run(int=0) final;
    bool finish() final;

  private:
    void print();
    void analyze();

  private:
    void register_vars();
    TString label(TString parName);
  private:
    std::vector<int> interested_vars;
    std::vector<double> center;
    std::vector<double> covariance;
    std::vector<std::string> corr_vars;
    std::vector<std::string> interested_names;
    TH1 *h = nullptr;
    TH2 *h2r = nullptr;
    TH2 *h2 = nullptr;
};
#endif
