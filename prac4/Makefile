
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

all:	reduction reduction_shuffle
reduction:	reduction.cu Makefile
		nvcc reduction.cu -o reduction $(INC) $(NVCCFLAGS) $(LIB)

reduction_shuffle:	reduction_shuffle.cu Makefile
		nvcc reduction_shuffle.cu -o reduction_shuffle $(INC) $(NVCCFLAGS) $(LIB)
clean:
		rm -f reduction

