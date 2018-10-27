#!/bin/bash

cd GooStats-release
if [ ! -d build_GooFit ]; then
  mkdir build_GooFit
fi
googletestVersion=1.8.1
if [ ! -f v${googletestVersion}.tar.gz ]; then
  wget https://github.com/google/googletest/archive/release-${googletestVersion}.tar.gz -O v${googletestVersion}.tar.gz
fi
if [ ! -d googletest ]; then
  tar zxvf v${googletestVersion}.tar.gz
  mv googletest-release-${googletestVersion} googletest
fi
if [ ! -d build_googletest ]; then
  mkdir build_googletest
fi
if [ ! -d build_GooStats ]; then
  mkdir build_GooStats
fi

cat >setup.sh<<EOF
#!/bin/bash
export GTEST_ROOT="$(readlink -f ../GooStats-release/googletest-install)"
export GOOFIT_DIR="$(readlink -f ../GooStats-release/GooFit-install)"
export CMAKE_PREFIX_PATH=\$GOOFIT_DIR:\$CMAKE_PREFIX_PATH
EOF
