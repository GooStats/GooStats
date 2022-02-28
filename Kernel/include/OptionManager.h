/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef OptionManager_H
#define OptionManager_H
#include <map>
#include <string>
#include "Database.h"
#include "Utility.h"
// protocol for option configset class
class OptionManager : public Database {
public:
  // can be fileName or key=value sentence
  template<class T = double>
  T getOrConvert(const std::string &key) const {
      if(has(key)) {
        auto v = GooStats::Utility::convert<T>(get(key));
        if(has<double>(key) && get<double>(key)!=v) throw GooStatsException("Inconsistent status!");
        return v;
      }
      return get<double>(key);
    }
  //! yes(key): if user forgot to put key, program will throw
  bool yes(const std::string &key) const;
  //! hasAndYes(key): if user forgot to put key, program return false
  bool hasAndYes(const std::string &key) const;
  void printAllOptions() const;
};
#endif
