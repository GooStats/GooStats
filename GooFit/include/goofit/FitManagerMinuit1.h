#ifndef FITMANAGER_MINUIT1_HH
#define FITMANAGER_MINUIT1_HH

//#include "goofit/rootstuff/TMinuit.h"
#include "TMinuit.h"
extern PdfBase* pdfPointer; 
extern int numPars; 

void FitFun(int &npar, double *gin, double &fun, double *fp, int iflag); 

class FitManager { 
public:
  FitManager (PdfBase* dat);
  ~FitManager ();
  void setMaxCalls (double mxc) {overrideCallLimit = mxc;}
  void setupMinuit ();
  void runMigrad (); 
  void fit (); 
  TMinuit* getMinuitObject () {return minuit;} 
  void getMinuitValues () const;
  static int getNumFreePars ();
  TMinuit* minuit; 
  static std::vector<Variable*> vars; 
  static bool minim_conv;
  static bool hesse_conv;
private:
  double overrideCallLimit; 
};

#endif 
