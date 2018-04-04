BUILD :=  ../../../build_GooStats
config := ${BUILD}/CMakeCache.txt
CMKFLS := $(shell find ../.. -name CMakeLists.txt)
all: config
	bash -c "cd ${BUILD}; make -j juno;"
	ln -sf ${BUILD}/Modules/naive-Juno/juno
debug: config
	bash -c "cd ${BUILD}; make ;"

clean: config
	bash -c "cd ${BUILD}/Modules/naive-Juno; make clean"

config: ${CMKFLS}
	bash -c 'cd ${BUILD}; cmake ../GooStats'
