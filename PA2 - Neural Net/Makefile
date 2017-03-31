# normally, I use cmake, but trying to keep example simple.

# if eigen3 is not in system search path
EIGENDIR =/usr/include/eigen3
EIGENOPT = 
# possible options
EIGENOPT += -DEIGEN_NO_DEBUG 
EIGENOPT += -fopenmp  # let eigen use multiple cores for matrix multiplication
#EIGENOPT += -DNDEBUG t

CCX = g++
CXXFlags += -O2 -Wall --std=c++11 -I$(EIGENDIR) $(EIGENOPT)
LDFlags += -lm

# On linux, we need to add -lrt library for the BenchTimer code 
LDFlags += -lrt


all : pa2

pa2 : pa2.o  pa2_fann.o pa_util.o noise.o nn_util.o nn_structs.o
	$(CCX) $(CXXFlags) $(LDFlags) -o pa2 pa2.o pa2_fann.o  pa_util.o noise.o nn_util.o nn_structs.o


pa2.o : pa2.cc pa2_fann.h 
	$(CCX) $(CXXFlags) -c  pa2.cc

pa2_fann.o : pa2_fann.cc pa2_fann.h pa_util.h
	$(CCX) $(CXXFlags) -c  pa2_fann.cc

pa_util.o : pa_util.cc pa_util.h
	$(CCX) $(CXXFlags) -c  pa_util.cc

nn_structs.o: nn_structs.cpp noise.o nn_util.o pa_util.o
	$(CCX) $(CXXFlags) -c nn_structs.cpp nn_util.o pa_util.o -o nn_structs.o

noise.o: noise.cpp noise.h
	$(CCX) $(CXXFlags) -c noise.cpp -o noise.o

nn_util.o: nn_util.cpp nn_util.h pa_util.o
	$(CCX) $(CXXFlags) -c nn_util.cpp pa_util.o -o nn_util.o

clean:
	rm  pa2 *.o
