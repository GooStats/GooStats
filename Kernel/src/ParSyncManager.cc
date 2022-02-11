/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ParSyncManager.h"
#include "GooStatsException.h"
#include <iostream>
#include "IDataManager.h"

IDataManager::strategy ParSyncManager::getStrategy(const std::string &name) const {
  if(!initialized) throw GooStatsException("ParSyncManager::init() should be called before using it");
  if(strategies.find(name)!=strategies.end()) 
    return strategies.at(name);
  else {
    return IDataManager::Current;
  }
}
IDataManager *ParSyncManager::chooseManager(const std::string &name,IDataManager *daughter) const {
  if(getStrategy(name)==IDataManager::Current) return daughter;
  try {
    while(!daughter->isResponsibleFor(getStrategy(name))) {
      if(!daughter->parent()) {
        std::cerr<<"["<<daughter->name()<<"] has no parent. fail to find the responsible configset."<<std::endl;
        throw GooStatsException("Has no parent");
      }
      daughter = daughter->parent();
    }
    return daughter;
  } catch ( std::out_of_range &ex ) {
    std::cerr<<"Looping through the hierarchy and cannot find the suitable parent.."<<std::endl;
    throw GooStatsException("Cannot find suitable configset");
  } catch (...) {
    std::cerr<<"strange exception .. "<<std::endl;
    throw GooStatsException("Cannot find suitable configset");
  }
}
const IDataManager *ParSyncManager::chooseManager(const std::string &name,const IDataManager *daughter) const {
  if(getStrategy(name)==IDataManager::Current) return daughter;
  try {
    while(!daughter->isResponsibleFor(getStrategy(name))) {
      if(!daughter->parent()) {
        std::cerr<<"["<<daughter->name()<<"] has no parent. fail to find the responsible configset."<<std::endl;
        throw GooStatsException("Has no parent");
      }
      daughter = daughter->parent();
    }
    return daughter;
  } catch ( std::out_of_range &ex ) {
    std::cerr<<"Looping through the hierarchy and cannot find the suitable parent.."<<std::endl;
    throw GooStatsException("Cannot find suitable configset");
  } catch (...) {
    std::cerr<<"strange exception .. "<<std::endl;
    throw GooStatsException("Cannot find suitable configset");
  }
}
