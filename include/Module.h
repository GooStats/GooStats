/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef Module_H
#define Module_H
// interface for the basic unit of GooStats Analysis program
#include <vector>
#include <string>
#include "Bit.h"
class Module {
  public:
    Module(const std::string &_name,Bit _bit,Bit _dep) :
      m_name(_name),bit(_bit),dep(_dep) { }
  public:
    virtual bool init() = 0;
    virtual bool run() = 0;
    virtual bool finalize() = 0;
  public:
    std::string name() const { return m_name; }
    Bit& category() { return bit; }
    Bit& dependence() { return dep; }
    const Bit& category() const { return bit; }
    const Bit& dependence() const { return dep; }
  protected:
    std::string m_name;
    Bit bit;
    Bit dep;
};
#endif
