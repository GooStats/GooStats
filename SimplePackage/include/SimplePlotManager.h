/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef SimplePlotManager_H
#define SimplePlotManager_H
#include "PlotManager.h"
class SimplePlotManager : public PlotManager {
 public:
  void draw(int event, const std::vector<DatasetManager *> &datasets) final;
  using PlotManager::draw;

 private:
  virtual std::map<std::string, std::vector<DatasetManager *>> groupByName(const std::vector<DatasetManager *> &);
};
#endif
