BUILD :=  ../../../build_GooStats
config := ${BUILD}/CMakeCache.txt
CMKFLS := $(shell find ../.. -name CMakeLists.txt)
all: config
	bash -c "cd ${BUILD}; make -j fit;"
	ln -sf ${BUILD}/Modules/simpleFit/fit

clean: config
	bash -c "cd ${BUILD}/Modules/simpleFit; make clean"

config: ${CMKFLS}
	bash -c 'cd ${BUILD}; cmake ../GooStats'
