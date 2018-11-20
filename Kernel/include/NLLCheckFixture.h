/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef NLLCheckFixture_H
#define NLLCheckFixture_H
#include "gtest/gtest.h"
#include <vector>
#include <string>
class NLLCheckFixture : public ::testing::Test { 
  public: 
    NLLCheckFixture();
    void SetUp();
    void TearDown();
    ~NLLCheckFixture();
  protected:
    void load_result(const std::string &path,const std::string &type,std::vector<double> &LLs,double &finalLL);
    std::vector<double> reference_LL;
    std::vector<double> new_LL;
    double reference_finalLL;
    double new_finalLL;
};
#endif
