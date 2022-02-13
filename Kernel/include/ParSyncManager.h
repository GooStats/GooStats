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

using Level = int;
class ParSyncManager {
  public:
    virtual ~ParSyncManager() = default;;
    virtual void init() {}
    virtual std::map<std::string, Level> getStrategies() { return m_strategies; }
  private:
    std::map<std::string, Level> m_strategies;
};
#endif
