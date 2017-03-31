//CS4793 - progex1 - prob #1 - A* implementation
//Written By: Justin Lye

#if !defined(__JEL_GRAPH_HEADER__)
#define __JEL_GRAPH_HEADER__
#include<ctime>
#include<stdexcept>
#include"state_list.h"

namespace jel {
	class graph {
	private:
		//manages open list priority queue and closed list queue
		jel::state_list* s_list;
		//constant to be loaded into the register
		int register_goal;
		//number of nodes expanded while searching for operation path
		long nodes_expanded;
		//time elapsed before path was found in seconds
		double runtime;
		//dequeues open list, enqueues closed list, if goal state is dequeued returns pointer to state otherwise the state is expanded and nullptr is returned
		jel::state* expand_to_goal();
		//inaccessible constructor
		
	public:
		//constructor - takes register_goal as integer arg
		graph(int);
		graph();
		//~graph() {
			//if (s_list != nullptr) { delete s_list; }
		//}

		//member methods
		//calls expand_to_goal until returned value is not nullptr
		void Search();

		//set desired register constant. must be greater than 0. this is the constant to be loaded.
		inline void SetRegisterConstant(int val) {
			if (val < 0) {
				throw std::runtime_error("register goal should be greater than or equal to 0. Use SetGoal(const int&) before searching.");
			} else {
				register_goal = val;
			}
		}

		
	};
}

#endif
