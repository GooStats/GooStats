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
#include <iostream>
#include "ParSyncManager.h"
#include "GooStatsException.h"
ParSyncManager *BasicManager::strategyManager = nullptr;
void BasicManager::adoptFather(IDataManager* father_) { 
  std::cout<<"["<<name()<<"] adopt ["<<father_->name()<<"] as its father."<<std::endl;
  m_father = father_; 
}
BasicManager *BasicManager::chooseManager(const std::string &key) {
  if(strategyManager)
    return static_cast<BasicManager*>(strategyManager->chooseManager(key,this));
  else
    throw GooStatsException("Please call BasicManager::setStrategyManager(..) before using any BasicManager for creating/refering/linking variables");
}
const BasicManager *BasicManager::chooseManager(const std::string &key) const {
  if(strategyManager)
    return static_cast<const BasicManager*>(strategyManager->chooseManager(key,this));
  else
    throw GooStatsException("Please call BasicManager::setStrategyManager(..) before using any BasicManager for creating/refering/linking variables");
}
Variable *BasicManager::createVar(const std::string &key,double val,double err,double min,double max) {
  return chooseManager(key)->m_impl.createVar(key,val,err,min,max);
}
Variable *BasicManager::linkVar(const std::string &key,const std::string &source) {
  return chooseManager(key)->m_impl.linkVar(key,source);
}
bool BasicManager::hasVar(const std::string &key) const {
  return chooseManager(key)->m_impl.hasVar(key);
}
Variable *BasicManager::var(const std::string &key) const {
  return chooseManager(key)->m_impl.var(key);
}
const std::string &BasicManager::varOwner(const std::string &key) const {
  return chooseManager(key)->name();
}
