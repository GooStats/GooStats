/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef BasicManager_H
#define BasicManager_H
#include <map>
#include <string>
#include <utility>
#include <vector>
using Level = int;
class Variable;
class BasicManager {
  public:
  explicit BasicManager(std::string name_);
  const std::string &name() const { return m_name; }

  static void setParSyncConfig(const std::map<std::string, int> &configs) { s_configs = configs; }
  static void dump();

  Variable *createVar(const std::string &key, double val, double err, double min, double max);
  Variable *linkVar(const std::string &key, const std::string &source);
  bool hasVar(const std::string &key) const;
  Variable *var(const std::string &key) const;

  private:
  std::string getKey(const std::string &key) const;
  const std::string m_name;
  std::vector<std::string> m_parents;
  static std::map<std::string, Level> s_configs;
  static std::map<std::string, std::shared_ptr<Variable>> s_vars;
  std::shared_ptr<Variable> getVariable(const std::string &key) const;
};
#endif
