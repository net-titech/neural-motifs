all: roi_align_kernel.cu roi_align_kernel.h
	nvcc -c -o roi_align.cu.o roi_align_kernel.cu --compiler-options -fPIC -gencode arch=compute_61,code=sm_61 -gencode arch=compute_60,code=sm_60
clean:
	rm roi_align.cu.o
