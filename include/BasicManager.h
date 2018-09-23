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
#include "IDataManager.h"
#include <string>
class Variable;
#include "BasicManagerImpl.h"
class ParSyncManager;
class BasicManager : public IDataManager {
  public:
    BasicManager(const std::string name_) : IDataManager(), m_name(name_), m_impl(name_) {};
    const std::string &name() const { return m_name; }
    virtual bool isResponsibleFor(strategy) const { return true; }
    void adoptParent(IDataManager* parent_);
    IDataManager* parent() const { return m_parent; }
  public:
    static void setStrategyManager(ParSyncManager *_s) { strategyManager = _s; }
    Variable *createVar(const std::string &key,double val,double err,double min,double max);
    Variable *linkVar(const std::string &key,const std::string &source);
    bool hasVar(const std::string &key) const;
    Variable *var(const std::string &key) const;
    const std::string &varOwner(const std::string &key) const;

  private:
    BasicManager *chooseManager(const std::string &key);
    const BasicManager *chooseManager(const std::string &key) const;
  private:
    static ParSyncManager* strategyManager;
    const std::string m_name;
    IDataManager *m_parent;
    BasicManagerImpl m_impl;
};
#endif
