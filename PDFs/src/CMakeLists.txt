cuda_add_more_library(GooStatsPDFs
  DarkNoiseConvolutionPdf.cu
  DumperPdf.cu
  GeneralConvolutionPdf.cu
  Mach4GGResponseFunctionPdf.cu
  Mach4GGResponseFunctionPdf_impl.cu
  HistogramPdf.cu
  MultiVariatePdf.cu
  PdfCache.cu
  RangedExpPdf.cu
  ProductPdf.cu
  SumLikelihoodPdf.cu
  SumPdf.cu)
target_link_libraries(GooStatsPDFs GOOFIT::goofit_lib)
