#!/bin/bash

# download GooStats
GooStatsVersion=5.1.0
if [ ! -f v${GooStatsVersion}.tar.gz ]; then
  wget https://github.com/GooStats/GooStats/archive/v${GooStatsVersion}.tar.gz
fi
if [ ! -d GooStats-${GooStatsVersion} ]; then
  tar zxvf v${GooStatsVersion}.tar.gz
  mv GooStats-${GooStatsVersion} GooStats
fi

# download googletest
googletestVersion=1.8.1
if [ ! -f v${googletestVersion}.tar.gz ]; then
  wget https://github.com/google/googletest/archive/release-${googletestVersion}.tar.gz -O v${googletestVersion}.tar.gz
fi
if [ ! -d googletest ]; then
  tar zxvf v${googletestVersion}.tar.gz
  mv googletest-release-${googletestVersion} googletest
fi

# make folder
if [ ! -d build_googletest ]; then
  mkdir build_googletest
fi
if [ ! -d build_GooFit ]; then
  mkdir build_GooFit
fi
if [ ! -d build_GooStats ]; then
  mkdir build_GooStats
fi

# make setup.sh
cat >setup.sh<<EOF
#!/bin/bash
export GTEST_ROOT="$(readlink -f googletest-install)"
export GOOFIT_DIR="$(readlink -f GooFit-install)"
export CMAKE_PREFIX_PATH=\$GOOFIT_DIR:\$CMAKE_PREFIX_PATH
EOF

ln -s GooStats/setup/compile.sh
