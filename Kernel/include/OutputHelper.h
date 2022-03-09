/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef OutputHelper_H
#define OutputHelper_H
#include <functional>
#include <string>
class InputManager;
#include <vector>
class OutputHelper {
 public:
  void registerTerm(const std::string &name, const std::function<double()> &);
  void flush();
  const std::vector<std::string> names() const { return m_names; }
  const std::vector<double *> addresses();
  double value(const std::string &n) const;

 private:
  std::vector<std::function<double()>> functions;
  std::vector<std::string> m_names;
  std::vector<double> values;
};
#endif
