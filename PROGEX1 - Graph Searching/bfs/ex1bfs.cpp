#include<iostream>
#include<ctime>
#include<stdexcept>
#include"graph.h"

int prompt();
void setvalue(int);
int main(int argc, char* argv[]) try {
	int val = prompt();
	while (val >= 0) {
		setvalue(val);
		val = prompt();
	}
	
	return 0;
} catch (std::exception& e) {
	std::cerr << e.what() << "\n\n";
	return 1;
} catch (...) {
	std::cerr << "unknown error occurred\n\n";
	return 1;
}

void setvalue(int val) {
	jel::graph g;
	jel::state s = g.FindTest(val);
	std::cout << "Runtime:\t" << g.Runtime() << "(s)\n\n";
	std::cout << "Path:   ";
	s.PrintHistory();
	std::cout << "\nNodes Expanded:\t" << g.NodesExpanded() << " Queue Size:\t" << g.QueueSize() << "\n\n";
	//std::cout << "\n\nCurrent Queue:\nState(value) : { path cost, parent op }\n\n" << g._queue << '\n';	

	
}


int prompt() {
	int val = 0;
	std::cout << "Enter desired register value: ";
	std::cin >> val;
	return val;
}

