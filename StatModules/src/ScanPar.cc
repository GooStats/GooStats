/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "ScanPar.h"
#include "InputManager.h"
#include "OutputManager.h"
#include "GSFitManager.h"
bool ScanPar::run(int) {
  if(!GlobalOption()->hasAndYes("ScanPar")) return true;
  auto parName = GlobalOption()->query("scanPar");
  auto id = getGSFitManager()->get_id(parName);
  auto left = std::stod(GlobalOption()->query("scanParMin"));
  auto right = std::stod(GlobalOption()->query("scanParMax"));
  auto Npoint = std::atoi(GlobalOption()->query("scanParN").c_str());
  auto gMinuit = getGSFitManager()->getFitManager()->getMinuitObject();

  gMinuit->FixParameter(id-1);
  gMinuit->SetPrintLevel(-1);
  int fI = gMinuit->fIstrat; // the strategy
  for(int i = 0;i<Npoint;++i) {
    double xx = left+(right-left)/(Npoint-1)*i;
    gMinuit->Command(Form("SET PARameter %d %lf",id,xx));
    if(i==1) gMinuit->Command("SET STRategy 0");
    getGSFitManager()->getFitManager()->runMigrad();
    //gMinuit->Migrad();
    //gMinuit->mnmigr(); // just minimize, doesn't calculate errors
    getGSFitManager()->getFitManager()->getMinuitValues();
    getGSFitManager()->eval();
    getOutputManager()->subFit(-1);
    printf("Scanning [%10s] (%3d/%3d;%5.2lf,%5.2lf,%5.2lf) -> %10.2lf\n",parName.c_str(),i,Npoint,xx,left,right,
	   getGSFitManager()->minus2lnlikelihood());
  }
  gMinuit->Command(Form("SET STRategy %d",fI));
  gMinuit->SetPrintLevel(0);
  gMinuit->Release(id-1);
  return true;
}
