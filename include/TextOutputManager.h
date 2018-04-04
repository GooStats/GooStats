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
using std::string;
class TextOutputManager {
  public:
    static string data(string name,double v,string unit);
    static const string rate(string name,double v,double e,double max,double min,string unit,bool penalty = false,double penalty_mean = 0,double penalty_sigma = 0) ;
    static const string qch(string name,double v,double e,double max,double min,bool penalty = false,double penalty_mean = 0,double penalty_sigma = 0) ;
    static unsigned int get_effective_digits(double x);
    static string show_numbers(double x,unsigned int d);
    static string prettify_speciesName(string name);
};
#endif
