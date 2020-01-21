/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef ParSyncManager_H
#define ParSyncManager_H
#include <string>
#include <map>
class IDataManager;
// class controlling the parameter syncrhonization strategies
#include "BasicManager.h"
#include "InputConfig.h"

class ParSyncManager {
  public:
    typedef IDataManager::strategy level;
  public:
    virtual ~ParSyncManager() {};
    virtual BasicManager *createParSyncSet(const InputConfig&) { return new BasicManager("default"); }
    virtual std::map<std::string, level> getStrategies() { return std::map<std::string,level>(); };
    void init() { strategies = getStrategies(); initialized = true; }
  private:
    friend class BasicManager;
    const IDataManager *chooseManager(const std::string &key,const IDataManager *daughter) const;
    IDataManager *chooseManager(const std::string &key,IDataManager *daughter) const;
  private:
    IDataManager::strategy getStrategy(const std::string &name) const;
    std::map<std::string, level> strategies;
    bool initialized = false;
};
#endif
