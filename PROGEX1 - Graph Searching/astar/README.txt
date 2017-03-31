CS 4793 - progex1 - Justin Lye - A* Implementation

HOW TO COMPILE:

Please use to following commands in sequence on CSX machine:

0. navigate to directory where source files are located (e.g. >cd progex1/astar)
1. >g++ -O2 -Wall --std=c++11 -c state_list.cpp
2. >g++ -O2 -Wall --std=c++11 -c graph.cpp state_list.o
3. >g++ -O2 -Wall --std=c++11 -c ex1astar.cpp graph.o state_list.o
4. >g++ -O2 -o ex1astar ex1astar.o graph.o state_list.o

If the program will not compile or crashes during runtime please
contact justin lye at jlye@okstate.edu.