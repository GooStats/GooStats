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
    auto &data = list<T>();
    if (data.find(key) == data.end() || !check) {
      data[key] = val;
    } else {
      throw GooStatsException("Duplicate key <" + key + "> insertion");
    }
  }

  template<class T = std::string>
  [[nodiscard]] bool has(std::string key) const {
    const auto &data = list<T>();
    return data.find(key) != data.end();
  }

  template<class T = std::string>
  T get(std::string key, bool check = true) const {
    const auto &data = list<T>();
    if (data.find(key) != data.end()) {
      return data.at(key);
    } else {
      if (check) throw GooStatsException("Key <" + key + "> not found");
      else
        return {};
    }
  }

  template<class T = std::string>
  [[nodiscard]] const std::map<std::string, T> &list() const = delete;

  template<class T = std::string>
  [[nodiscard]] std::map<std::string, T> &list() = delete;

#define DECLARE_TYPE(T, VAR)                                                                                           \
public:                                                                                                                \
  template<>                                                                                                           \
  [[nodiscard]] const std::map<std::string, T> &list<T>() const {                                                      \
    return VAR;                                                                                                        \
  };                                                                                                                   \
                                                                                                                       \
private:                                                                                                               \
  template<>                                                                                                           \
  [[nodiscard]] std::map<std::string, T> &list<T>() {                                                                  \
    return VAR;                                                                                                        \
  };                                                                                                                   \
                                                                                                                       \
private:                                                                                                               \
  std::map<std::string, T> (VAR);

  DECLARE_TYPE(double, m_double);
  DECLARE_TYPE(int, m_int);
  DECLARE_TYPE(std::string, m_str);
};

#endif//BX_GOOSTATS_DATABASE_H
