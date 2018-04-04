
goofit_add_library(Variable
    Variable.cc
    )

goofit_add_library(FitManager
    FitManager.cc
    )

goofit_add_library(Faddeeva
    Faddeeva.cc
    )

goofit_add_library(DataSet
    DataSet.cc
    )

goofit_add_library(BinnedDataSet
    BinnedDataSet.cc
    )

goofit_add_library(UnbinnedDataSet
    UnbinnedDataSet.cc
    )

goofit_add_library(FunctorWriter
    FunctorWriter.cc
    )

goofit_add_library(FitControl
    FitControl.cc
    )


goofit_add_library(PdfBase
    PdfBase.cc
    )


set(lib_list
    Variable Faddeeva DataSet
    BinnedDataSet UnbinnedDataSet
    FitManager FunctorWriter FitControl
    PdfBase
    )

add_library(goofit_base INTERFACE)
target_link_libraries(goofit_base INTERFACE ${lib_list})
install(TARGETS goofit_base ${lib_list}
  EXPORT "${TARGETS_EXPORT_NAME}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
