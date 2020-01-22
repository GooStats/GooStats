#!/bin/bash
cd build_GooFit
cmake ../GooStats/GooFit -DCMAKE_INSTALL_PREFIX="$(cd ..; pwd)/GooFit-install"
make -j8
make -j8 install
cd ../build_googletest
cmake ../googletest -DCMAKE_INSTALL_PREFIX="$(cd ..; pwd)/googletest-install"
make -j8
make -j8 install
cd ..
