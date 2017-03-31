//CS4793 - progex1 - prob #1 - A* implementation
//Written By: Justin Lye

#include<iostream>
#include<stdexcept>
#include"graph.h"

void titlePrompt();
void inputPrompt();

int main(int argc, char* argv[]) try {
	int target = 0;  //variable to hold user input
	titlePrompt();   //print program title
	inputPrompt();   //prompt user to enter a value
	std::cin >> target; //capture value using standard input
	jel::graph g;
	while (target >= 0) {
		g.SetRegisterConstant(target);
		g.Search();
		inputPrompt();    //prompt user to enter a value
		std::cin >> target; //capture value using standard input
	}
	return 0;
} catch (std::exception& e) {
	std::cerr << e.what() << std::endl;
	return 1;
} catch (...) {
	std::cerr << "Unknown error has occurred." << std::endl;
	return 1;
}

//print title when program starts
void titlePrompt() {
	std::cout << "A* implementation - Justin Lye 10/06/2016\n\n";
	std::cout << "To quit, enter a negative value when prompted for input\n\n";
}

//prompt the user for input
void inputPrompt() {
	std::cout << "Enter an integer value: ";
}