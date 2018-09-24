/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef IDataManager_H
#define IDataManager_H
#include <memory>
class Variable;
// Synchronized parameter sets
class IDataManager {
  public:
    enum strategy { Global, Period, HalfSphere, Current};
    virtual const std::string &name() const = 0;
    virtual bool isResponsibleFor(strategy theStrategy) const = 0;
    virtual void adoptParent(IDataManager* parent_) = 0;
    virtual IDataManager* parent() const = 0;
    virtual Variable *createVar(const std::string &key,double val,double err,double min,double max) = 0;
    virtual Variable *linkVar(const std::string &key,const std::string &source) = 0;
    virtual bool hasVar(const std::string &key) const = 0;
    virtual Variable *var(const std::string &key) const = 0;
    virtual const std::string &varOwner(const std::string &key) const = 0;
    virtual ~IDataManager() {};
};
#define information() Form("%s : line %d",__FILE__, __LINE__)
#endif
