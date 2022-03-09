/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef BatchOutputManager_H
#define BatchOutputManager_H
#include <memory>
#include <vector>

#include "Module.h"
#include "TTree.h"
class SumLikelihoodPdf;
struct Variable;
class TFile;
class BatchOutputManager : public Module {
 public:
  BatchOutputManager() : Module("BatchOutputManager") {}
  virtual bool init() override;
  virtual bool run(int event) override;
  virtual bool finish() override;
  bool check() const final { return has("OutputManager") && has("InputManager"); }

 public:
  void fill_rates();
  virtual void flush_txt(std::ostream &, std::map<std::string, double> &) const;
  void bindAllParameters(const SumLikelihoodPdf *pdf_);

 protected:
  void cd();
  std::shared_ptr<TTree> tree;
  std::vector<Variable *> vars;

 private:
  void flush_tree();

 private:
  const SumLikelihoodPdf *pdf = nullptr;
  /**
     *  \defgroup TTree binding for saving fit output
     *  @{
     */
 public:
  void bind(const std::string &brName);
  void fill(const std::string &brName, double value) { results[brName][nfit] = value; }

 private:
  int nfit = 0;
  std::map<std::string, double[200]> results;
  /**@}*/
};
#endif
