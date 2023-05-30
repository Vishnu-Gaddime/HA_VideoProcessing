#######################################################################################################################################
#
#	Basic Makefile for SDAccel 2017.1
#	Lorenzo Di Tucci, Emanuele Del Sozzo
#	{lorenzo.ditucci, emanuele.delsozzo}@polimi.it
#	Usage make [emulation | build | clean | clean_sw_emu | clean_hw_emu | clean_hw | cleanall] TARGET=<sw_emu | hw_emu | hw>
#
#
#######################################################################################################################################
VCC=v++
CC=g++

PROJECT_HOME=../../../..
KERNEL_OPTIMIZATION_LEVEL=--optimize s
HOST_OPTIMIZATION_LEVEL=-O3
INCLUDE=-I ${PROJECT_HOME}/include/

SRC=$(PROJECT_HOME)/src/fpga/
HOST_SRC=$(SRC)/main_cl.c
HOST_HDRS=

# conditionally setup compiler flags based on whether or not Xilinx environment is set
ifdef XILINX_XRT
HOST_CFLAGS=-D FPGA_DEVICE -D C_KERNEL -D CREATE_FPGA_KERNEL -Wall -I${XILINX_XRT}/include/ ${INCLUDE} -pthread -Wl,-no-as-needed ${HOST_OPTIMIZATION_LEVEL}
HOST_LFLAGS=-L${XILINX_XRT}/lib/ -lOpenCL -lrt -lstdc++ -lpthread -lm
else
HOST_CFLAGS=-Wall -pthread -Wl,-no-as-needed ${HOST_OPTIMIZATION_LEVEL} -fopenmp ${INCLUDE}
HOST_LFLAGS=-lpthread -lm
endif

#name of host executable
HOST_EXE=host.exe

N_JOBS=4

#kernel
KERNEL_SRC=$(SRC)/vadd.cpp
KERNEL_HDRS=$(SRC)
KERNEL_FLAGS= ${KERNEL_OPTIMIZATION_LEVEL} ${INCLUDE}
KERNEL_EXE=vadd.xclbin
KERNEL_NAME=vadd

#custom flag to give to xocc
KERNEL_LDCLFLAGS= --config link.ini
# KERNEL_LDCLFLAGS=--connectivity.nk $(KERNEL_NAME):4\
# 	--xp param:compiler.preserveHlsOutput=1 \
# 	--hls.max_memory_ports $(KERNEL_NAME) \
# 	--hls.memory_port_data_width $(KERNEL_NAME):512
KERNEL_ADDITIONAL_FLAGS=--advanced.param compiler.acceleratorBinaryContent=dcp

TARGET_DEVICE=xilinx_u250_gen3x16_xdma_2_1_202010_1

#TARGET for compilation [sw_emu | hw_emu | hw]
TARGET=hw
REPORT_FLAG=n
REPORT=
ifeq (${TARGET}, sw_emu)
$(info software emulation)
TARGET=sw_emu
ifeq (${REPORT_FLAG}, y)
$(info creating REPORT for software emulation set to true. This is going to take longer at it will synthesize the kernel)
REPORT=--report estimate
else
$(info I am not creating a REPORT for software emulation, set REPORT_FLAG=y if you want it)
REPORT=
endif
else ifeq (${TARGET}, hw_emu)
$(info hardware emulation)
TARGET=hw_emu
REPORT=--report estimate
else ifeq (${TARGET}, hw)
$(info system build)
TARGET=hw
REPORT=--report system
else
$(info no TARGET selected)
endif

PERIOD:= :
UNDERSCORE:= _

ifndef XILINX_XRT
$(info Xilinx enviroment is not set. Only building host)
all: host
else
all: host emconfig xclbin
endif

clean: clean_logs
	rm -rf *.xo *.exe *.out *.xclbin* *json *csv *log *summary _x .run .Xil .ipcache *.jou sw_emu hw_emu hw emconfig.json *.o *.vtk common host_dep

clean_logs:
	rm -rf *log*
clean_sw_emu: clean
	rm -rf sw_emu
clean_hw_emu: clean
	rm -rf hw_emu
clean_hw: clean
	rm -rf hw

cleanall: clean_sw_emu clean_hw_emu clean_hw
	rm -rf _xocc_* xcl_design_wrapper_*

check_TARGET:
ifeq (${TARGET}, none)
	$(error Target can not be set to none)
endif

host: check_TARGET $(HOST_SRC) $(HOST_HDRS) common host_dep
	${CC} ${HOST_SRC} ${HOST_HDRS} ${HOST_CFLAGS} common/*.o host_dep/*.o -o ${HOST_EXE} ${HOST_LFLAGS}

common:
	[ ! -d common ] && mkdir common
	@for f in $(shell ls ${PROJECT_HOME}/src/common); do ${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} -c $(SRC)/common/$${f} -o common/$${f}.o; done

host_dep:
	[ ! -d host_dep ] && mkdir host_dep
	@for f in $(shell ls ${PROJECT_HOME}/src/host_dep); do ${CC} ${HOST_CFLAGS} ${HOST_LFLAGS} -c ${SRC}/host_dep/$${f} -o host_dep/$${f}.o; done

xo:	check_TARGET
	$(VCC) --platform $(TARGET_DEVICE) --target $(TARGET) --j $(N_JOBS) --compile --include $(KERNEL_HDRS) --save-temps $(REPORT) --kernel $(KERNEL_NAME) $(KERNEL_SRC) $(KERNEL_LDCLFLAGS) $(KERNEL_FLAGS) --output $(KERNEL_NAME).xo

xclbin: check_TARGET xo
	$(VCC) --platform $(TARGET_DEVICE) --target $(TARGET) -R2 --link $(KERNEL_ADDITIONAL_FLAGS) --optimize s --j $(N_JOBS) --include $(KERNEL_HDRS) --save-temps $(REPORT) --kernel $(KERNEL_NAME) $(KERNEL_NAME).xo $(KERNEL_FLAGS) $(KERNEL_LDCLFLAGS) --output $(KERNEL_EXE)
