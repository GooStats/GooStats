goofit_add_library(
  GooStatsPDFs
  DarkNoiseConvolutionPdf.cu
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
  SumLikelihoodPdf.cu)
target_link_libraries(GooStatsPDFs PUBLIC GooFit::PDFs)
target_include_directories(
  GooStatsPDFs PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/PDFs/include>"
                      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

goofit_add_library(
  GooStatsPDFs_NLL
  DarkNoiseConvolutionPdf.cu
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
  SumLikelihoodPdf.cu)
target_link_libraries(GooStatsPDFs_NLL PUBLIC PDFs_NLL)
target_include_directories(
  GooStatsPDFs_NLL
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/PDFs/include>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
