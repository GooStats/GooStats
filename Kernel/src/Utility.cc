/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "Utility.h"
#include <map>
namespace GooStats {
  namespace Utility {
    // naive splitter
    std::string strip(const std::string &k) {
      auto pool = k;
      auto pComment = pool.find("//");
      if(pComment==std::string::npos) pComment = pool.find("#");
      if(pComment!=std::string::npos) pool = pool.substr(0,pComment);
      auto i0 = pool.find_first_not_of("\t ");
      auto in = pool.find_last_not_of("\t ");
      return i0!=std::string::npos ? pool.substr(i0,in-i0+1) : std::string();
    }
    std::vector<std::string> splitter(std::string source, std::string flag) {
      std::vector<std::string> result;
      while (source.find(flag) != std::string::npos) {
        auto position = source.find(flag);
        result.push_back(strip(source.substr(0, position)));
        source = source.substr(position + 1, source.size() - position - 1);
      }
      if (!source.empty()) result.push_back(strip(source));
      return result;
    }
    std::string escape(const std::string &str, std::string purge, std::string underscore,
                       std::vector<std::string> full) {
      std::string result(str);
      std::map<std::string, std::string> pairs = {{underscore, "_"}, {purge, ""}};
      for (auto p: pairs)
        for (auto c: p.first) {
          while (result.find(c) != std::string::npos) {
            result = result.substr(0, result.find(c)) + p.second + result.substr(result.find(c) + 1);
          }
        }
      for (auto word: full) {
        while (result.find(word) != std::string::npos) {
          result = result.substr(0, result.find(word)) + result.substr(result.find(word) + word.length());
        }
      }
      return result;
    }
  }// namespace Utility
}// namespace GooStats
