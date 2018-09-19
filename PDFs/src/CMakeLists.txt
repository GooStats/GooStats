cuda_add_more_library(GooStatsPDFs
  DarkNoiseConvolutionPdf.cu
  DumperPdf.cu
  GeneralConvolutionPdf.cu
  ResponseFunctionPdf.cu
  ResponseFunctionPdf_impl.cu
  ResponseFunctionPdf_NLRES.cu
  HistogramPdf.cu
  IntegralInsideBinPdf.cu
  MultiVariatePdf.cu
  PdfCache.cu
  PullPdf.cu
  RangedExpPdf.cu
  ProductPdf.cu
  SumLikelihoodPdf.cu
  SumPdf.cu)
target_link_libraries(GooStatsPDFs GOOFIT::goofit_lib)
