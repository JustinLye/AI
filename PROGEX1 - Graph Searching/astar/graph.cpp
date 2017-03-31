//CS4793 - progex1 - prob #1 - A* implementation
//Written By: Justin Lye

#include"graph.h"

jel::graph::graph() : s_list(nullptr), register_goal(0), nodes_expanded(0), runtime(0) {}

//graph constructor takes register_goal as argument
jel::graph::graph(int init_goal) : s_list(nullptr), register_goal(init_goal), nodes_expanded(0), runtime(0) {
	if (register_goal < 0) {
		throw std::runtime_error("register goal should be greater than or equal to 0. Use SetGoal(const int&) before searching.");
	}
}

//dequeues open list, enqueues closed list, if goal state is dequeued returns pointer to state otherwise the state is expanded and nullptr is returned
jel::state* jel::graph::expand_to_goal() {
	jel::state& current_state = s_list->top(); //dequeue state with lowest cost. note this also enqueues the closed list
	int reg_val = current_state.RegisterValue(); //copy register value to local variable for readability
	if (reg_val == register_goal) {
		return &current_state; //check for goal state
	}
	nodes_expanded++; //increment expansion counter
	if (reg_val % 3 == 0) { //check if register value is divisable by 3
		s_list->enqueue( //enqueue new state to open list with division edge
			jel::state(
				reg_val / 3,
				register_goal,
				jel::op::DIV,
				&current_state)
		);
	} else {
		s_list->enqueue(//enqueue new state to open list with double edge
			jel::state(
				reg_val * 2,
				register_goal,
				jel::op::DUB,
				&current_state)
		);
		if (reg_val > 1) {
			s_list->enqueue( //enqueue new state to open list with addition edge
				jel::state(
					reg_val + 1,
					register_goal,
					jel::op::ADD,
					&current_state)
			);
			s_list->enqueue( //enqueue new state to open list with subtraction edge
				jel::state(
					reg_val - 1,
					register_goal,
					jel::op::SUB,
					&current_state)
			);
		}
	}
	return nullptr; //return nullptr (goal has not be dequeued/found)
}

void jel::graph::Search() try {
	s_list = new jel::state_list; //allocate space for lists'
	s_list->enqueue(jel::state(1, register_goal, jel::op::ONE, nullptr)); //insert root of graph/tree
	jel::state* goal_state = nullptr; //set sentinel pointer to null
	std::clock_t begin = clock(); //start runtime clock
	do {
		goal_state = expand_to_goal(); //seach for operation path
	} while ((goal_state == nullptr)); //stop when expand_to_goal returns pointer other than nullptr
	std::clock_t end = clock(); //stop runtime clock
	runtime = double(end - begin) / CLOCKS_PER_SEC; //load time elapsed into runtime variable

	//print results
	std::cout << "\n-------------------------------\nConstant Loaded\n\n\tTime required:\t " << runtime << "(s)\n\n";
	std::cout << "\tNodes Expanded:\t" << nodes_expanded << "\n\n\tOPERATION Sequence:\n\n";
	goal_state->PrintHistory();
	std::cout << "\n\n\tOPEN List:\n\n";
	s_list->PrintOpenList();
	std::cout << "\n\n\tCLOSED List:\n\n";
	s_list->PrintClosedList();
	std::cout << "\n\n";

	delete s_list; //release memory
	s_list = nullptr; //set to null

} catch (std::exception& e) {
	if (s_list != nullptr) {
		delete s_list;
		s_list = nullptr;
	}
	throw e;
} catch (...) {
	if (s_list != nullptr) {
		delete s_list;
		s_list = nullptr;
	}
	throw std::runtime_error("error: the register could not be loaded due to an unknown exception");
}