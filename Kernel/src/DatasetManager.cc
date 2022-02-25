/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "DatasetManager.h"
#include "DatasetController.h"
#include "GooStatsException.h"
#include "goofit/PDFs/GooPdf.h"

#define DEFINE_DatasetManager(T, var, ZERO)                                                                            \
  template<>                                                                                                           \
  void DatasetManager::set<T>(const std::string &_name, T x) {                                                         \
    if (var.find(_name) == var.end()) {                                                                                \
      var[_name] = x;                                                                                                  \
    } else {                                                                                                           \
      std::cout << fullName() << " Warning: duplicate request to insert terms <" << _name << ">" << std::endl;         \
    }                                                                                                                  \
  }                                                                                                                    \
  template<>                                                                                                           \
  T DatasetManager::get<T>(const std::string &_name, bool checkEmpty) {                                                \
    if (var.find(_name) != var.end()) {                                                                                \
      return var.at(_name);                                                                                            \
    } else {                                                                                                           \
      if (checkEmpty) {                                                                                                \
        std::cout << fullName() << " Warning: request non-existed terms <" << _name << ">" << std::endl;               \
        gSystem->StackTrace();                                                                                         \
      }                                                                                                                \
      return ZERO;                                                                                                     \
    }                                                                                                                  \
  }                                                                                                                    \
  template<>                                                                                                           \
  T DatasetManager::get<T>(const std::string &_name, bool checkEmpty) const {                                          \
    if (var.find(_name) != var.end()) {                                                                                \
      return var.at(_name);                                                                                            \
    } else {                                                                                                           \
      if (checkEmpty) {                                                                                                \
        std::cout << fullName() << " Warning: request non-existed terms <" << _name << ">" << std::endl;               \
        gSystem->StackTrace();                                                                                         \
      }                                                                                                                \
      return ZERO;                                                                                                     \
    }                                                                                                                  \
  }                                                                                                                    \
  template<>                                                                                                           \
  bool DatasetManager::has<T>(const std::string &_name) const {                                                        \
    return var.find(_name) != var.end();                                                                               \
  }

DEFINE_DatasetManager(std::string, m_str, "");
DEFINE_DatasetManager(int, m_int, 0);
DEFINE_DatasetManager(bool, m_bool, false);
DEFINE_DatasetManager(double, m_double, 0);
DEFINE_DatasetManager(Variable *, m_var, nullptr);
DEFINE_DatasetManager(PdfBase *, m_pdf, nullptr);
DEFINE_DatasetManager(std::vector<std::string>, m_components, std::vector<std::string>());
DEFINE_DatasetManager(std::vector<double>, m_coeff, std::vector<double>());
DEFINE_DatasetManager(std::vector<Variable *>, m_vars, std::vector<Variable *>());
DEFINE_DatasetManager(std::vector<PdfBase *>, m_pdfs, std::vector<PdfBase *>());
DEFINE_DatasetManager(BinnedDataSet *, m_bindata, nullptr);

void DatasetManager::setLikelihood(GooPdf *pdf) { likelihood = std::shared_ptr<GooPdf>(pdf); }
