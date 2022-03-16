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
    // naive split
    extern std::string strip(const std::string &);
    extern std::vector<std::string> split(std::string source, std::string flag);
    extern std::string escape(const std::string &str,
                              std::string purge = ")",
                              std::string underscore = "(",
                              std::vector<std::string> full = {"default.", "global."});
    extern BinnedDataSet *toDataSet(RawSpectrumProvider *, Variable *, const std::string &, bool check_e0 = true);
    extern void save(RawSpectrumProvider *, const std::string &name, TH1 *);
    template <typename T>
    T convert(const std::string &) = delete;
    template <>
    double convert<double>(const std::string &v);
    template <>
    int convert<int>(const std::string &v);

    /// original source: https://stackoverflow.com/a/26221725/8732904
    /// by iFreilicht@StackOverflow under CC0 1.0
    template <typename... Args>
    std::string string_format(const std::string &format, Args... args) {
      int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
      if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
      }
      auto size = static_cast<size_t>(size_s);
      auto buf = std::unique_ptr<char[]>(new char[size]);
      std::snprintf(buf.get(), size, format.c_str(), args...);
      return {buf.get(), buf.get() + size - 1};  // We don't want the '\0' inside
    }
  }  // namespace Utility
}  // namespace GooStats
#endif
