/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef BasicManagerImpl_H
#define BasicManagerImpl_H
#include <string>
#include <map>
#include <memory>
struct Variable;
class BasicManagerImpl {
  private:
    BasicManagerImpl(const std::string name_) : m_name(name_),m_var() {}
    Variable *createVar(const std::string &key,double val,double err,double min,double max);
    Variable *linkVar(const std::string &key,const std::string &source);
    bool hasVar(const std::string &key) const;
    Variable *var(const std::string &key) const;
    void dump(const std::string &indent) const;
    std::string m_name;
    std::map<std::string, std::shared_ptr<Variable> > m_var;
    friend class BasicManager;
};
#endif
