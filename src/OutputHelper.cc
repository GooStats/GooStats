/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "OutputHelper.h"
#include "GooStatsException.h"
void OutputHelper::registerTerm(const std::string &name, const std::function<double(InputManager*)>& function) {
  m_names.push_back(name);
  functions.push_back(function);
  values.push_back(0);
}
void OutputHelper::flush(InputManager *inputManager) {
  for(size_t i = 0;i<functions.size();++i) {
    values.at(i) = functions.at(i)(inputManager);
  }
}
const std::vector<double*> OutputHelper::addresses() {
  std::vector<double*> addr;
  for(size_t i = 0;i<values.size();++i) {
    addr.push_back( & ( values[i] ) );
  }
  return addr;
}
double OutputHelper::value(const std::string &n) {
  // remember to flush
  for(size_t i = 0;i<m_names.size();++i) {
    if(m_names.at(i)==n) return values.at(i);
  }
  throw GooStatsException("key <"+n+"> not found");
}
