/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef BatchOutputManager_H
#define BatchOutputManager_H
#include "TTree.h"
#include <memory>
#include <vector>
class GooPdf;
class Variable;
class TFile;
class BatchOutputManager {
  public:
    virtual bool init();
    virtual bool run();
    virtual bool finish();
    void flush_tree();
    virtual void flush_txt(std::ostream &,std::map<std::string,double> &) const;

    void setOutputFileName(const std::string &n) { outName = n; }
    template<typename T>
      void bind(const std::string &brName,T *addr) {
	tree->Branch(brName.c_str(), addr);
      }
    void bindAllParameters(GooPdf *pdf);
  protected:
    void fill();
    void bindTree();
    std::shared_ptr<TTree> tree;
    //TTree* tree = nullptr;
    std::vector<Variable*> vars;
    TFile *out = nullptr;
    std::string outName;
};
#endif
