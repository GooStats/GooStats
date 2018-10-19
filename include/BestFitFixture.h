/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef BestFitFixture_H
#define BestFitFixture_H
#include "gtest/gtest.h"
#include <vector>
#include <string>
class BestFitFixture : public ::testing::Test { 
  public: 
    BestFitFixture();
    void SetUp();
    void TearDown();
    ~BestFitFixture();
    void setEntry(int entry_,int sub) { entry = entry_; subEntry = sub; }
  protected:
    void load_result(const std::string &path,std::vector<double> &out,std::vector<double> &rough_out);
    std::vector<double> reference_fit;
    std::vector<double> new_fit;
    std::vector<double> rough_reference_fit;
    std::vector<double> rough_new_fit;
    std::vector<std::string> species;
    std::vector<std::string> rough_species;
    int entry = 0;
    int subEntry = 0;
};
#endif
