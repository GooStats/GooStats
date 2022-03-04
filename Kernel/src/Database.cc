/*****************************************************************************/
// Author: Xuefeng Ding <xuefengd@princeton.edu>
// Institute: Department of Physics, Princeton University
// Date: 2/28/22
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2022 copyrighted.
/*****************************************************************************/

#include "Database.h"

#define DEFINE_METHOD(T, VAR)                                                                                          \
  template<>                                                                                                           \
  [[nodiscard]] const std::map<std::string, T> &Database::list<T>() const {                                            \
    return VAR;                                                                                                        \
  }                                                                                                                    \
  template<>                                                                                                           \
  [[nodiscard]] std::map<std::string, T> &Database::list<T>() {                                                        \
    return VAR;                                                                                                        \
  }

EXPAND_MACRO(DEFINE_METHOD);