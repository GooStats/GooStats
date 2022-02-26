/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef Utility_H
#define Utility_H
#include <string>
#include <vector>
#include "goofit/Variable.h"
class BinnedDataSet;
class RawSpectrumProvider;
class Variable;
class TH1;
namespace GooStats {
  namespace Utility {
    // naive splitter
    extern std::string strip(const std::string &);
    extern std::vector<std::string> splitter(std::string source, std::string flag);
    extern std::string escape(const std::string &str,std::string purge=")",std::string underscore="(",std::vector<std::string> full = {"default.","global."});
    extern BinnedDataSet *toDataSet(RawSpectrumProvider *, Variable *, const std::string &, bool check_e0 = true);
    extern void save(RawSpectrumProvider *, const std::string &name, TH1 *);
  }
}
#endif
