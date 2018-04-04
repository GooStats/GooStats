include_directories(
    ${PROJECT_SOURCE_DIR}/src/PDFs
    )

get_property(goo_extra GLOBAL PROPERTY GOOFIT_EXTRA_PDFS)

if(CUDA_SEPARABLE_COMPILATION)
    goofit_add_library(PDFs
        FitControl.cu
        GooPdf.cu
        PdfBase.cu
	ExpPdf.cu
        ${goo_extra})
else()
  message(FATAL_ERROR "You must use separable compilation (CUDA_SEPARABLE_COMPILATION=ON)")
endif()

target_link_libraries(PDFs goofit_base)
install(TARGETS PDFs
  EXPORT "${TARGETS_EXPORT_NAME}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
