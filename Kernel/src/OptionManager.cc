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

#include <iostream>

#include "GooStatsException.h"

void OptionManager::printAllOptions() const {
  std::cout << "*********Dump options parsed*********************************" << std::endl;
  for (const auto &pair : list<std::string>())
    std::cout << "" << (pair.first) << " => <" << (pair.second) << ">" << std::endl;
  std::cout << "-------------------------------------------------------------" << std::endl;
  for (const auto &pair : list<double>())
    std::cout << "" << (pair.first) << " => double <" << (pair.second) << ">" << std::endl;
  std::cout << "-------------------------------------------------------------" << std::endl;
  for (const auto &pair : list<int>())
    std::cout << "" << (pair.first) << " => int <" << (pair.second) << ">" << std::endl;
  std::cout << "*************************************************************" << std::endl;
}

bool OptionManager::yes(const std::string &key) const {
  return (get<std::string>(key) == "yes" || get<std::string>(key) == "on" || get<std::string>(key) == "true");
}

bool OptionManager::hasAndYes(const std::string &key) const { return has<std::string>(key) && yes(key); }
