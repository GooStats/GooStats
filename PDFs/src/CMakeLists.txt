goofit_add_library(GooStatsPDFs
  DarkNoiseConvolutionPdf.cu
  GeneralConvolutionPdf.cu
  GooStatsSumPdf.cu
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
)
if(NLL_CHECK)
  target_link_libraries(GooStatsPDFs PUBLIC GooStatsNLLCheck)
endif()
target_link_libraries(GooStatsPDFs PUBLIC GooFit::PDFs)
target_include_directories(GooStatsPDFs PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/PDFs/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
