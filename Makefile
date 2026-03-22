NVCC = /usr/local/cuda/bin/nvcc
FLAGS = -O3 -arch=sm_90 -std=c++17 --ptxas-options=-v

benchmark: benchmark.cu
	$(NVCC) $(FLAGS) -o $@ $<

sass: benchmark
	/usr/local/cuda/bin/cuobjdump --dump-sass $< > benchmark.sass

ptx: benchmark.cu
	$(NVCC) $(FLAGS) -ptx -o benchmark.ptx $<

clean:
	rm -f benchmark benchmark.ptx benchmark.sass ncu_output.txt
