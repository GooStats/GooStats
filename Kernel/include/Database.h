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
#include <iostream>
#include <map>
#include <string>

#include "GooStatsException.h"
#include "Utility.h"

class Database {
 public:
  template<class T>
  void set(std::string key, T val, bool check = true) {
    // don't use = delete
    // = delete will throw error at linking stage, not easy for debugging.
    // static_assert will throw error at compiling stage.
    static_assert(!std::is_same<T, const char *>::value,
                  "const char* type explicitly deleted due to high probability "
                  "of wrong usage. use std::string() to "
                  "wrap your values!");
    setImpl(key,val,check);
  }
  template<>
  void set<std::string>(std::string key, std::string val, bool check) {
    std::string type = "string";
    if(key.find(":")!=std::string::npos) {
      auto head = GooStats::Utility::split(key, ":");
      if (head.size() != 2) {
        std::cout << "Cannot split <" << key << ">" << std::endl;
        throw GooStatsException("illegal format");
      }
      key = head.at(0);
      type = head.at(1);
    }
    if(type=="double") {
      setImpl<double>(key, std::stod(val), check);
    } else if(type=="int") {
      setImpl<int>(key,std::stoi(val),check);
    } else if(type=="string") {
      setImpl<std::string>(key,val,check);
    } else {
      std::cerr<<"Unkown type <"<<type<<">"<<std::endl;
      throw GooStatsException("illegal type");
    }
  }

  template <class T = std::string>
  [[nodiscard]] bool has(std::string key) const {
    const auto &data = list<T>();
    return data.find(key) != data.end();
  }

  template <class T = std::string>
  T get(std::string key, bool check = true) const {
    const auto &data = list<T>();
    if (data.find(key) != data.end()) {
      return data.at(key);
    } else {
      if (check)
        throw GooStatsException("Key <" + key + "> not found");
      else
        return {};
    }
  }

  template <class T = std::string>
  [[nodiscard]] const std::map<std::string, T> &list() const = delete;

 private:
  template <class T>
  void setImpl(std::string key, T val, bool check = true) {
    auto &data = list<T>();
    if (data.find(key) == data.end() || !check) {
      if (has<T>(key)) {
        std::cerr << "Warning: duplicate key [" << key << "] found. old [" << get<T>(key) << "] new [" << val << "]"
                  << std::endl;
      }
      data[key] = val;
    } else {
      throw GooStatsException("Duplicate key <" + key + "> insertion");
    }
  }

  template <class T = std::string>
  [[nodiscard]] std::map<std::string, T> &list() = delete;

#define EXPAND_MACRO(MACRO) \
  MACRO(double, m_double);  \
  MACRO(int, m_int);        \
  MACRO(std::string, m_str);

#define DECLARE_TYPE(T, VAR) std::map<std::string, T>(VAR);

  EXPAND_MACRO(DECLARE_TYPE);
};

#define DECLARE_METHOD(T, VAR)                                             \
  template <>                                                              \
  [[nodiscard]] const std::map<std::string, T> &Database::list<T>() const; \
  template <>                                                              \
  [[nodiscard]] std::map<std::string, T> &Database::list<T>();

EXPAND_MACRO(DECLARE_METHOD);

#endif  //BX_GOOSTATS_DATABASE_H
