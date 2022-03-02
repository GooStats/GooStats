/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "BasicManager.h"
#include "GooStatsException.h"
#include "ParSyncManager.h"
#include "goofit/Variable.h"
#include <iostream>
#include <utility>

std::map<std::string, Level> BasicManager::s_configs;
std::map<std::string, std::shared_ptr<Variable>> BasicManager::s_vars;

Variable *BasicManager::createVar(const std::string &key, double val, double err, double min, double max) {
  auto finalKey = getKey(key);
  if (!hasVar(finalKey)) {
    std::shared_ptr<Variable> var_(std::make_shared<Variable>(finalKey, val, err, min, max));
    std::cout << "Inserting [" << key << "] to [" << m_name << "] => [" << finalKey << "]" << std::endl;
    s_vars.insert(std::make_pair(finalKey, var_));
  } else {
    std::cerr << "Create a var that already exists: [" << key << "] to [" << m_name << "] => [" << finalKey << "]"
              << std::endl;
  }
  return s_vars.at(finalKey).get();
}
Variable *BasicManager::linkVar(const std::string &key, const std::string &source) {
  auto finalKey = getKey(key);
  s_vars.insert(std::make_pair(finalKey, getVariable(source)));
  return s_vars.at(finalKey).get();
}
bool BasicManager::hasVar(const std::string &key) const {
  auto finalKey = getKey(key);
  return s_vars.find(finalKey) != s_vars.end();
}
Variable *BasicManager::var(const std::string &key) const { return getVariable(key).get(); }
std::shared_ptr<Variable> BasicManager::getVariable(const std::string &key) const {
  try {
    auto finalKey = getKey(key);
    return s_vars.at(finalKey);
  } catch (const GooStatsException &ex) {
    dump();
    throw ex;
  }
}
void BasicManager::dump() {
  for (const auto &pair: s_vars) {
    std::cout << pair.first << " " << pair.second << " " << pair.second->value << " ± " << pair.second->error << " ["
              << pair.second->lowerlimit << "," << pair.second->upperlimit << "] fixed? "
              << (pair.second->fixed ? "yes" : "no") << std::endl;
  }
}
std::string BasicManager::getKey(const std::string &key) const {
  if (s_configs.find(key) == s_configs.end()) { return m_name + "." + key; }
  auto level = s_configs.at(key);
  return (level < m_parents.size() ? m_parents.at(level) : m_name) + "." + key;
}
BasicManager::BasicManager(std::string name_) : m_name(std::move(name_)) {
  m_parents.emplace_back("global");
  auto pos = m_name.find('.');
  while (pos != std::string::npos) {
    m_parents.push_back(m_name.substr(0, pos));
    pos = m_name.find('.', pos + 1);
  }
}
