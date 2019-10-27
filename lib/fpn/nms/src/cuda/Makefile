all: nms_kernel.cu nms_kernel.h
	nvcc -c -o nms.cu.o nms_kernel.cu --compiler-options -fPIC -gencode arch=compute_61,code=sm_61 -gencode=arch=compute_60,code=sm_60
clean:
	rm nms.cu.o
