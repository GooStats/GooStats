/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#include "SimpleOutputBuilder.h"
#include "OutputHelper.h"

#include "goofit/PDFs/GooPdf.h"
#include "InputManager.h"
#include "BatchOutputManager.h"
#include "SumLikelihoodPdf.h"
#include "SumPdf.h"
#include "TMath.h"
#include "PlotManager.h"
void SimpleOutputBuilder::registerOutputTerms(OutputHelper *outHelper, InputManager *inputManager) {
  for(auto dataset : inputManager->Datasets()) {
    double exposure = dataset->get<double>("exposure");
    outHelper->registerTerm(dataset->name()+".exposure", [=](InputManager *) -> double {
      return exposure;
    });
  }
  outHelper->registerTerm("chi2", [](InputManager *inputManager) -> double {
    GooPdf *pdf = inputManager->getTotalPdf();
    pdf->setFitControl(new BinnedChisqFit);
    pdf->copyParams();
    return pdf->calculateNLL();
  });
  outHelper->registerTerm("NDF", [](InputManager *inputManager) -> double {
    SumLikelihoodPdf *pdf = static_cast<SumLikelihoodPdf*>(inputManager->getTotalPdf());
    int NDF = 0;
    for(auto component : pdf->Components()) {
      SumPdf *pdf = dynamic_cast<SumPdf*>(component);
      if(pdf) NDF += pdf->NDF();
    }
    return NDF;
  });
  outHelper->registerTerm("likelihood", [](InputManager *inputManager) -> double {
    GooPdf *pdf = inputManager->getTotalPdf();
    pdf->setFitControl(new BinnedNllFit);
    pdf->copyParams();
    return pdf->calculateNLL();
  });
  //outHelper->registerTerm("LLp", [](InputManager *inputManager) -> double { return 0; });
  //outHelper->registerTerm("LLpErr", [](InputManager *inputManager) -> double { return 0; });
}

void SimpleOutputBuilder::bindAllParameters(BatchOutputManager *writer,OutputHelper* outHelper) {
  auto addrs = outHelper->addresses();
  for(size_t i = 0;i<outHelper->names().size();++i) {
    writer->bind(outHelper->names().at(i), addrs.at(i));
  }
}
void SimpleOutputBuilder::flushOstream(BatchOutputManager *batchOut,OutputHelper *outHelper,std::ostream &out) {
  std::map<std::string,double> goodness;
  goodness["chi2"] = outHelper->value("chi2");
  goodness["chi2/NDF"] = goodness["chi2"]/outHelper->value("NDF");
  goodness["p-value"] = TMath::Prob(goodness["chi2"],int(outHelper->value("NDF")+0.5));
  goodness["likelihood"] = outHelper->value("likelihood");
  //goodness["LLp"] = outHelper->value("LLp");
  //goodness["LLpErr"] = outHelper->value("LLpErr");
  batchOut->flush_txt(out,goodness);
}
void SimpleOutputBuilder::draw(PlotManager *plot,InputManager *in) {
  plot->draw(in->Datasets());
}
