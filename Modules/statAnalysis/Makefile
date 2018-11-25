BUILD :=  ../../../build_GooStats
config := ${BUILD}/CMakeCache.txt
CMKFLS := $(shell find ../.. -name CMakeLists.txt)
all: config
	bash -c "cd ${BUILD}; make -j stat;"
	ln -sf ${BUILD}/Modules/statAnalysis/stat

clean: config
	bash -c "cd ${BUILD}/Modules/statAnalysis; make clean"

config: ${CMKFLS}
	bash -c 'cd ${BUILD}; cmake ../GooStats'
