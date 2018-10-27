/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef GSFitManager_H
#define GSFitManager_H
#include "goofit/FitManager.h"
#include "Module.h"
#include <memory>
#include "ToyMCLikelihoodEvaluator.h"
#include "TMath.h"
class InputManager;
class SumLikelihoodPdf;
class GSFitManager : public Module {
  public:
    GSFitManager() : Module("GSFitManager") { }
    bool run(int = 0) final;
    bool check() const final { return has("InputManager"); }
    FitManager *getFitManager() const { return fitManager.get(); }
    void restoreFitControl();
  public:
    double chi2() const { return m_chi2; }
    int NDF() const { return m_NDF; }
    double Prob() const { return TMath::Prob(chi2(),NDF()); }
    double minus2lnlikelihood() { return m_likelihood*2; }
    const std::vector<double> LLs() const { return toyMC.getLLs(); }
    double LLp() const { return m_LLp; }
    double LLpErr() const { return m_LLpErr; }
    bool minim_conv() const { return m_minim_conv; }
    bool hesse_conv() const { return m_hesse_conv; }
    bool LLfit() const;
  private:
    void eval();
  private:
    InputManager *getInputManager();
    const InputManager *getInputManager() const;
    SumLikelihoodPdf *sumpdf();
  private:
    ToyMCLikelihoodEvaluator toyMC;
    std::shared_ptr<FitManager> fitManager;
    bool m_minim_conv = false;
    bool m_hesse_conv = false;
    double m_likelihood;
    double m_chi2;
    int m_NDF;
    double m_LLp = -1;
    double m_LLpErr = 1e300;
};
#endif
