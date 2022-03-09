/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ANALYSIS_MANAGER_H
#define ANALYSIS_MANAGER_H
class Module;
#include <list>
#include <memory>
class AnalysisManager {
 public:
  AnalysisManager();
  bool registerModule(Module *module);
  bool hasModule(const std::string &name) const;
  virtual bool init();
  virtual bool run(int event = 0);
  virtual bool finish();
  bool checkGPU() const;

 private:
  std::list<std::shared_ptr<Module> > modules;
};
#define information() Form("%s : line %d", __FILE__, __LINE__)
#endif
