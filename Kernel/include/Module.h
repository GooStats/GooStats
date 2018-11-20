/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef Module_H
#define Module_H
// interface for the basic unit of GooStats Analysis program
#include <map>
#include <string>
#include "GooStatsException.h"
class Module {
  public:
    Module(const std::string &_name) : m_name(_name) { }
    virtual ~Module() { }
    const std::string &name() const { return m_name; }
  public:
    virtual bool preinit() { validate(); return true; }
    virtual bool init() { return true; }
    virtual bool run(int = 0/*event*/) { return true; }
    virtual bool finish() { return true; }
    virtual bool postfinish() { return true; }
    virtual bool check() const { return true; } // check dependences
  public:
    void validate() { if(!check()) throw GooStatsException(name()+": depdences check failed"); }
    void registerDependence(Module *dep,const std::string label="") { dependences[label==""?dep->name():label] = dep; }
    bool has(const std::string &mod) const { return dependences.find(mod)!=dependences.end(); }
    const Module *find(const std::string &mod) const { return dependences.at(mod); }
    Module *find(const std::string &mod) { return dependences.at(mod); }
    const std::string list() const;
  protected:
    std::string m_name;
    std::map<std::string,Module*> dependences;
};
#endif
