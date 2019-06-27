#!/bin/bash
cd build_GooFit
cmake ../GooStats/GooFit -DCMAKE_INSTALL_PREFIX="$(readlink -f ../GooFit-install)"
make -j8
make -j8 install
cd ../build_googletest
cmake ../googletest -DCMAKE_INSTALL_PREFIX="$(readlink -f ../googletest-install)"
make -j8
make -j8 install
cd ..
