/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Institute: Department of Physics, Princeton University
// Date: 2/28/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/

#ifndef BX_GOOSTATS_DATABASE_H
#define BX_GOOSTATS_DATABASE_H
#include "GooStatsException.h"
#include <map>
#include <string>

class Database {
public:
  template<class T = std::string>
  void set(std::string key, T val, bool check = true) {
    static_assert(!std::is_same<T, const char *>::value, "const char* type explicitly deleted due to high probability "
                                                         "of wrong usage. use std::string() to "
                                                         "wrap your values!");
    auto &data = store<T>::data;
    if (data.find(key) == data.end() || !check) {
      data[key] = val;
    } else {
      throw GooStatsException("Duplicate key <"+key+"> insertion");
    }
  }

  template<class T = std::string>
  bool has(std::string key) const {
    const auto &data = store<T>::data;
    return data.find(key) != data.end();
  }

  template<class T = std::string>
  T get(std::string key, bool check = true) const {
    const auto &data = store<T>::data;
    if (data.find(key) != data.end()) {
      return data.at(key);
    } else {
      if (check) throw GooStatsException("Key <"+key+"> not found");
      else
        return {};
    }
  }

  template<class T = std::string>
  std::map<std::string, T> list() const {
    return store<T>::data;
  };

private:
  template<class T>
  struct store {
    static std::map<std::string, T> data;
  };
};

template<typename T>
std::map<std::string, T> Database::store<T>::data = {};

#endif//BX_GOOSTATS_DATABASE_H
