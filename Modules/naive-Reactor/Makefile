BUILD :=  ../../../build_GooStats
config := ${BUILD}/CMakeCache.txt
CMKFLS := $(shell find ../.. -name CMakeLists.txt)
all: config
	bash -c "cd ${BUILD}; make -j reactor;"
	ln -sf ${BUILD}/Modules/naive-Reactor/reactor
debug: config
	bash -c "cd ${BUILD}; make ;"

clean: config
	bash -c "cd ${BUILD}/Modules/naive-Reactor; make clean"

config: ${CMKFLS}
	bash -c 'cd ${BUILD}; cmake ../GooStats'
