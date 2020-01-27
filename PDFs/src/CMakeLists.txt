goofit_add_library(GooStatsPDFs
  DarkNoiseConvolutionPdf.cu
  DumperPdf.cu
  GeneralConvolutionPdf.cu
  HistogramPdf.cu
  IBDPdf.cu
  IntegralInsideBinPdf.cu
  MultiVariatePdf.cu
  NeutrinoOscillationPdf.cu
  PdfCache.cu
  ProductPdf.cu
  PullPdf.cu
  ExpPullPdf.cu
  RatioPullPdf.cu
  RangedExpPdf.cu
  PoissonPullPdf.cu
  ReactorSpectrumPdf.cu
  ResponseFunctionPdf.cu
  ResponseFunctionPdf_NLRES.cu
  ResponseFunctionPdf_impl.cu
  SumLikelihoodPdf.cu
  SumPdf.cu)
if(NLL_CHECK)
  target_link_libraries(GooStatsPDFs GooStatsNLLCheck)
endif()
target_link_libraries(GooStatsPDFs GOOFIT::goofit_lib AllGPU)
