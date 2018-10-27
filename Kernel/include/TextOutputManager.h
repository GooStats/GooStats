/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef TEXT_OUTPUT_MANAGER_H
#define TEXT_OUTPUT_MANAGER_H
#include <string>
class TextOutputManager {
  public:
    static std::string data(std::string name,double v,std::string unit);
    static const std::string rate(std::string name,double v,double e,double max,double min,bool penalty = false,double penalty_mean = 0,double penalty_sigma = 0) ;
    static const std::string qch(std::string name,double v,double e,double max,double min,bool penalty = false,double penalty_mean = 0,double penalty_sigma = 0) ;
    static unsigned int get_effective_digits(double x);
    static std::string show_numbers(double x,unsigned int d);
    static std::string prettify_speciesName(std::string name);
    static void set_unit(const std::string &unit) { m_Unit = unit; }
    static const std::string &get_unit() { return m_Unit; }
  private:
    static const std::string rate(std::string name,double v,double e,double max,double min,std::string unit,bool penalty = false,double penalty_mean = 0,double penalty_sigma = 0) ;
    static std::string m_Unit;
};
#endif
