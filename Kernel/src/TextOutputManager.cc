/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "TextOutputManager.h"

#include <cmath>
#include <sstream>

#include "GooStatsException.h"
#include "TString.h"
std::string TextOutputManager::m_Unit = "day#timeskt";
std::string TextOutputManager::data(std::string name, double v, std::string unit) {
  std::ostringstream output;
  output.str("");
  output << name << " from " << v << " " << unit;
  return output.str();
}
const std::string TextOutputManager::rate(std::string name_,
                                          double v,
                                          double e,
                                          double max,
                                          double min,
                                          bool penalty,
                                          double penalty_mean,
                                          double penalty_sigma) {
  return rate(name_, v, e, max, min, m_Unit, penalty, penalty_mean, penalty_sigma);
}
const std::string TextOutputManager::rate(std::string name_,
                                          double v,
                                          double e,
                                          double max,
                                          double min,
                                          std::string unit,
                                          bool penalty,
                                          double penalty_mean,
                                          double penalty_sigma) {
  if (v > 8640 && unit == "day#times100t")
    return rate(name_,
                v / 86400,
                e / 86400,
                max / 86400,
                min / 86400,
                "sec#times100t",
                penalty,
                penalty_mean / 86400,
                penalty_sigma / 86400);
  if (v > 8640 && unit == "day#timeskt")
    return rate(name_,
                v / 86400,
                e / 86400,
                max / 86400,
                min / 86400,
                "sec#timeskt",
                penalty,
                penalty_mean / 86400,
                penalty_sigma / 86400);
  unit = unit == "" ? unit : "(" + unit + ")^{-1}";
  std::string name = prettify_speciesName(name_);
  std::ostringstream output;
  output.str("");
  if (e > 0) {
    if (((max - v) / (max - min) > 0.01) && (v - min) / (max - min) > 0.01) {
      unsigned int effective_digits = get_effective_digits(e);
      output << name << " = " << show_numbers(v, effective_digits) << " #pm " << show_numbers(e, effective_digits)
             << " " << unit;
    } else
      output << name << " = " << show_numbers(v, 2) << " " << unit << " [Railed]";
    if (penalty) {
      unsigned int ed = get_effective_digits(penalty_sigma);
      output << " [p] " << show_numbers(penalty_mean, ed) << " #pm " << show_numbers(penalty_sigma, ed);
    }
  } else if (e == 0)
    output << name << " = " << v << " " << unit << " [Fixed]";
  else {
    throw GooStatsException("Error cannot be less than zero");
  }
  return output.str();
}
const std::string TextOutputManager::qch(std::string name_,
                                         double v,
                                         double e,
                                         double max,
                                         double min,
                                         bool penalty,
                                         double penalty_mean,
                                         double penalty_sigma) {
  return rate(name_, v, e, max, min, "", penalty, penalty_mean, penalty_sigma);
}
unsigned int TextOutputManager::get_effective_digits(double x) {
  unsigned int i = 0;
  while (fabs(x) < 1) {
    x *= 10;
    ++i;
    if (i >= 6)
      break;
  }
  return i + 1;
}
std::string TextOutputManager::show_numbers(double x, unsigned int d) { return Form(Form("%%.%dlf", d), x); }
std::string TextOutputManager::prettify_speciesName(std::string orig) {
  if (orig.find(".") != std::string::npos)
    orig = orig.substr(orig.find_last_of(".") + 1);
  const static std::string digits("0123456789");
  static std::string letters;
  if (!letters.size()) {
    for (char c = 'a'; c <= 'z'; c++) {
      letters += c;
      letters += toupper(c);
    }
  }
  unsigned start = 0, end = 0;
  std::string result;
  do {
    if (isalpha(orig[start])) {
      std::string symbol, massno;
      end = orig.find_first_not_of(letters, start);
      symbol = orig.substr(start, end - start);
      start = end;
      if (start < orig.size()) {
        if (isdigit(orig[start])) {
          end = orig.find_first_not_of(digits, start);
          massno = orig.substr(start, end - start);
        } else if (orig.size() > start + 1 && orig[start] == '-' && isdigit(orig[start + 1])) {
          ++start;
          end = orig.find_first_not_of(digits, start);
          massno = orig.substr(start, end - start);
        }
      }
      if (massno.size())
        result += std::string("^{") + massno + "}";
      if (symbol == "nu")
        result += "#";
      result += symbol;
    } else {
      end = orig.find_first_of(letters, start);
      result += orig.substr(start, end - start);
    }
    start = end;
  } while (start < orig.size());
  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i] == '_' && (i == result.size() - 1 || result[i + 1] != '{'))
      result[i] = ' ';
  }
  return result;
}
