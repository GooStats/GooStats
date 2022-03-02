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

#include "BatchOutputManager.h"
#include "GSFitManager.h"
#include "InputManager.h"
#include "PlotManager.h"
#include "TMath.h"
#include "goofit/FitManager.h"
#include "goofit/PDFs/GooPdf.h"
void SimpleOutputBuilder::registerOutputTerms(OutputHelper *outHelper, InputManager *inputManager,
                                              GSFitManager *gsFitManager) {
  for (auto dataset: inputManager->Datasets()) {
    if (dataset->has<double>("exposure")) {
      double exposure = dataset->get<double>("exposure");
      outHelper->registerTerm(dataset->fullName() + ".exposure", [=]() -> double { return exposure; });
    } else {
      std::cerr<<"Warning: exposure not found in ["<<dataset->fullName()<<"]"<<std::endl;
    }
  }
  outHelper->registerTerm("chi2", [gsFitManager]() -> double { return gsFitManager->chi2(); });
  outHelper->registerTerm("NDF", [gsFitManager]() -> double { return gsFitManager->NDF(); });
  outHelper->registerTerm("likelihood", [gsFitManager]() -> double { return gsFitManager->minus2lnlikelihood() / 2; });
  outHelper->registerTerm("minim_conv", [gsFitManager]() -> double { return gsFitManager->minim_conv(); });
  outHelper->registerTerm("hesse_conv", [gsFitManager]() -> double { return gsFitManager->hesse_conv(); });
  outHelper->registerTerm("LLp", [gsFitManager]() -> double { return gsFitManager->LLp(); });
  outHelper->registerTerm("LLpErr", [gsFitManager]() -> double { return gsFitManager->LLpErr(); });
}

void SimpleOutputBuilder::bindAllParameters(BatchOutputManager *batch, OutputHelper *outHelper) {
  auto addrs = outHelper->addresses();
  for (size_t i = 0; i < outHelper->names().size(); ++i) {
    if (outHelper->names().at(i).substr(0, 9) != "_forEval_") batch->bind(outHelper->names().at(i));
  }
}
void SimpleOutputBuilder::fillAllParameters(BatchOutputManager *batch, OutputHelper *outHelper) {
  for (auto name: outHelper->names())
    if (name.substr(0, 9) != "_forEval_") batch->fill(name, outHelper->value(name));
}
void SimpleOutputBuilder::flushOstream(BatchOutputManager *batchOut, OutputHelper *outHelper, std::ostream &out) {
  goodness["chi2"] = outHelper->value("chi2");
  goodness["chi2/NDF"] = goodness["chi2"] / outHelper->value("NDF");
  goodness["p-value"] = TMath::Prob(goodness["chi2"], int(outHelper->value("NDF") + 0.5));
  goodness["likelihood"] = outHelper->value("likelihood");
  goodness["LLp"] = outHelper->value("LLp");
  goodness["LLpErr"] = outHelper->value("LLpErr");
  batchOut->flush_txt(out, goodness);
}
void SimpleOutputBuilder::draw(int event, GSFitManager *gsFitManager, PlotManager *plot, InputManager *in) {
  plot->draw(event, in->Datasets());
  plot->drawLikelihoodpValue(event, goodness["likelihood"], gsFitManager->LLs());
}
