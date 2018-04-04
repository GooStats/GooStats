/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ConfigsetManager.h"
#include "GooStatsException.h"
const OptionManager *ConfigsetManager::optionManager() const {
  if(!m_optionManager) {
    throw GooStatsException("Query options before calling ConfigsetManager::setOptionManager()");
  } else 
    return m_optionManager.get();
}
OptionManager *ConfigsetManager::optionManager() {
  if(!m_optionManager) {
    throw GooStatsException("Query options before calling ConfigsetManager::setOptionManager()");
  } else 
    return m_optionManager.get();
}
