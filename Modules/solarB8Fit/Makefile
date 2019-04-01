BUILD :=  ../../../build_GooStats
config := ${BUILD}/CMakeCache.txt
CMKFLS := $(shell find ../.. -name CMakeLists.txt)
all: config
	bash -c "cd ${BUILD}; make -j solarB8Fit;"
	ln -sf ${BUILD}/Modules/solarB8Fit/solarB8Fit

clean: config
	bash -c "cd ${BUILD}/Modules/solarB8Fit; make clean"

config: ${CMKFLS}
	bash -c 'cd ${BUILD}; cmake ../GooStats'
