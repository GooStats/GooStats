/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef DatasetManagers_H
#define DatasetManagers_H
#include <iostream>
#include <map>
/*! \class DatasetDelegate
 *  \brief Delegate protocol for DatasetManager class. A controller of
 *  DatasetManager will be called when initialize its components
 */
class DatasetManager;
class GooPdf;
class PdfBase;
/*! \class DatasetManager
 *  \brief Manager class of dataset, the basic unit exposed to GooFit. A dataset
 *  correspond to one piece of likelihood. It can be more specific dataset: a
 *  specrtrumdataset, with spectrum, components, exposure etc. or a pull-tem
 *  dataset. A pull term data set can be of rate or generally on anything, or a
 *  pull term on the relationship between terms.
 *
 *  The datasetmanager is desgined to take observer pattern, and usually
 *  multiple datasetmanager will listen to one common configsetmanager.
 */
#include <memory>
#include <string>
#include <utility>
#include <vector>
struct Variable;
class GooPdf;
class BinnedDataSet;
class DatasetController;
class DatasetManager {
 public:
  DatasetManager(std::string name_, std::string configName_)
      : controller(nullptr), m_configName(std::move(configName_)), m_name(std::move(name_)){};
  virtual ~DatasetManager() = default;
  [[nodiscard]] std::string fullName() const { return m_configName + "." + m_name; }
  [[nodiscard]] const std::string &name() const { return m_name; }
  [[nodiscard]] const std::string &configName() const { return m_configName; }
  void setController(DatasetController *_d) { controller = _d; }
  DatasetController *getController() { return controller; }
  void setLikelihood(GooPdf *);
  [[nodiscard]] const GooPdf *getLikelihood() const { return likelihood.get(); }
  GooPdf *getLikelihood() { return likelihood.get(); }
  template <typename T>
  void set(const std::string &, T, bool = true) = delete;
  template <typename T>
  T get(const std::string &, bool = true) const = delete;
  template <typename T>
  T get(const std::string &, bool = true) = delete;
  template <typename T>
  bool has(const std::string &) const = delete;
  bool hasAndYes(const std::string &key) const;

 private:
  std::shared_ptr<GooPdf> likelihood;
  DatasetController *controller{};
  const std::string m_configName;
  const std::string m_name;
  std::map<std::string, std::string> m_str;
  std::map<std::string, int> m_int;
  std::map<std::string, bool> m_bool;
  std::map<std::string, double> m_double;
  std::map<std::string, Variable *> m_var;
  std::map<std::string, PdfBase *> m_pdf;
  std::map<std::string, std::vector<std::string>> m_components;
  std::map<std::string, std::vector<double>> m_coeff;
  std::map<std::string, std::vector<Variable *>> m_vars;
  std::map<std::string, std::vector<PdfBase *>> m_pdfs;
  std::map<std::string, BinnedDataSet *> m_bindata;
};
#define DECLARE_DatasetManager(T)                            \
  template <>                                                \
  void DatasetManager::set<T>(const std::string &, T, bool); \
  template <>                                                \
  T DatasetManager::get<T>(const std::string &, bool) const; \
  template <>                                                \
  T DatasetManager::get<T>(const std::string &, bool);       \
  template <>                                                \
  bool DatasetManager::has<T>(const std::string &) const;

DECLARE_DatasetManager(std::string);
DECLARE_DatasetManager(int);
DECLARE_DatasetManager(bool);
DECLARE_DatasetManager(double);
DECLARE_DatasetManager(Variable *);
DECLARE_DatasetManager(PdfBase *);
DECLARE_DatasetManager(std::vector<std::string>);
DECLARE_DatasetManager(std::vector<double>);
DECLARE_DatasetManager(std::vector<Variable *>);
DECLARE_DatasetManager(std::vector<PdfBase *>);
DECLARE_DatasetManager(BinnedDataSet *);
#endif
