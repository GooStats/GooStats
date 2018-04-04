/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef OptionManager_H
#define OptionManager_H
#include <string>
// protocol for option manager class
class OptionManager {
  public:
    virtual bool parse(const std::string &fileName) = 0;
    virtual std::string query(const std::string &key) const = 0;
    virtual bool has(const std::string &key) const = 0;
    //! yes(key): if user forgot to put key, program will throw
    virtual bool yes(const std::string &key) const = 0;
    //! hasAndYes(key): if user forgot to put key, program return false
    virtual bool hasAndYes(const std::string &key) const = 0;
};
#endif
