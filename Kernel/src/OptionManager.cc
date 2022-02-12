/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "OptionManager.h"
#include "GooStatsException.h"
#include <iostream>

template<typename T = std::string>
T convert(const std::string &v) { return v; }

template<>
double convert<double>(const std::string &v) {
  return std::stod(v);
}

template<typename T>
T OptionManager::get(const std::string &key, bool thr) const {
  if (m_data.find(key) == m_data.end()) {
    if (thr) throw GooStatsException(key + " not found");
    else
      return T();
  }
  return convert<T>(m_data.at(key));
}


bool OptionManager::has(const std::string &key) const {
  return m_data.find(key) != m_data.end();
}

void OptionManager::printAllOptions() const {
  std::cout << "*********Dump options parsed*********************************" << std::endl;
  for (const auto &pair: m_data)
    std::cout << "" << (pair.first) << " => <" << (pair.second) << ">" << std::endl;
  std::cout << "*************************************************************" << std::endl;
}

bool OptionManager::yes(const std::string &key) const {
  return (get(key) == "yes" || get(key) == "on" || get(key) == "true");
}

bool OptionManager::hasAndYes(const std::string &key) const {
  return has(key) && yes(key);
}

void OptionManager::set(const std::string &key, const std::string &value, bool allowOverwrite) {
  if (m_data.find(key) != m_data.end() && !allowOverwrite) {
    std::cout << "Duplicate term <" << key << "> found"
              << " old: <" << m_data.at(key) << "> -> new: <" << value << ">" << std::endl;
  }
  m_data[key] = value;
}

template
double OptionManager::get<double>(const std::string &, bool) const;