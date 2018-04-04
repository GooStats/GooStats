/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef PlotManager_H
#define PlotManager_H
#include <map>
#include <string>
#include <vector>
#include "Rtypes.h"
class TObject;
class DatasetManager;
class GooPdf;
class TH1;
class TF1;
class SumPdf;
class TFile;
class TCanvas;
class PlotManager {
  public:
    virtual bool init();
    virtual bool run();
    virtual bool finish();

  protected:
    struct Config {
      int color; ///< color
      int style; ///< line style , 1 is solid line
    };
    class TF1Helper {
      public:
	TF1Helper(GooPdf *pdf_,double norm_ = 1);
	TF1 *getTF1() { return f; }
      private:
	double eval(double *xx,double *par);
	TH1 *data; // root will delete it
	double norm;
      private:
	TF1 *f; // root will delete it
    };
  public:
    virtual void draw(const std::vector<DatasetManager*> &datasets);
    virtual TCanvas *drawSingleGroup(const std::string &name,const std::vector<DatasetManager*> &datasets) ;
    //! user can set the color and line style of each components by specifying the names
    //! plot() will look up the name of components of SumPdf
    //! by default different colors will be used for each species and all solid lines
    virtual void draw(SumPdf *pdf,std::map<std::string,Config> config= std::map<std::string,Config>());
    void setOutputFileName(const std::string &o) { _outName = o; }
  protected:
    EColor getColor(const std::string &n) const {
      return colorlibrary.find(n)!=colorlibrary.end()?colorlibrary.at(n):kBlack;
    }
    virtual void set_gStyle();
    void cd();
    bool createPdf() const { return _createPdf; };
    const std::string &outName() const { return _outName; };
    std::vector<TObject*> toBeSaved;
  private:
    TFile *out = nullptr;
    std::string _outName;
    bool _createPdf = true;
    //! configuration for plooting all species
    std::map<std::string, EColor> colorlibrary {
      {"kRed", kRed},
	{"kOrange", kOrange},
	{"kYellow", kYellow},
	{"kGreen", kGreen},
	{"kBlue", kBlue},
	{"kViolet", kViolet},
	{"kAzure", kAzure},
	{"kWhite", kWhite},
	{"kBlack", kBlack},
	{"kGray", kGray},
	{"kMagenta", kMagenta},
	{"kCyan", kCyan},
	{"kSpring", kSpring},
	{"kTeal", kTeal},
	{"kPink", kPink},
	{"kBlack", kBlack},
    };
};
#endif
