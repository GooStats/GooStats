/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef IDataManager_H
#define IDataManager_H
#include <memory>
struct Variable;
// Synchronized parameter sets
class IDataManager {
  public:
  virtual const std::string &name() const = 0;
  virtual Variable *createVar(const std::string &key, double val, double err, double min, double max) = 0;
  virtual Variable *linkVar(const std::string &key, const std::string &source) = 0;
  virtual bool hasVar(const std::string &key) const = 0;
  virtual Variable *var(const std::string &key) const = 0;
  virtual ~IDataManager(){};
};
#endif
