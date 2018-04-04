/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef RawSpectrumProvider_H
#define RawSpectrumProvider_H
#include <map>
#include <string>
// Protocol for raw spectrum provider
class RawSpectrumProvider {
  public:
    bool registerSpecies(const std::string &name,int n_,const double *real_,double e0_,double de_);
    bool registerPeak(const std::string &name,double peakE_);
    bool registerComplexSpecies(const std::string &name,const std::map<std::string,double> &br_);
    int n(const std::string &name) const;
    double const* pdf(const std::string &name) const;
    double e0(const std::string &name) const; // in keV
    double de(const std::string &name) const; // in keV
    double peakE(const std::string &name) const; // in keV
    double br(const std::string &name,std::string &subName) const; // branching ratio
  private:
    std::map<std::string, int> n_map;
    std::map<std::string, double const*> real_map;
    std::map<std::string, double> e0_map;
    std::map<std::string, double> de_map;
    std::map<std::string, double> peakE_map;
    std::map<std::string, std::map<std::string, double> > br_map;
};
#endif
