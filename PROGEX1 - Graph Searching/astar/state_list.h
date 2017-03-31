//CS4793 - progex1 - prob #1 - A* implementation
//Written By: Justin Lye

//state_list object implements A* algorithm

#if !defined(__JEL_STATELIST_HEADER__)
#define __JEL_STATELIST_HEADER__

#include<vector>
#include<algorithm>
#include<cmath>
#include<stdexcept>
#include<limits>
#include"state.h"
//#include"discard_queue.h"
namespace jel {
	class state_list {
	private:
		//jel::discard_queue trash_bin; //The intention of this member is to ensure memory is released on destruction. Any allocation of memory by a state_list method should be followed by a call to the trash_bin.insert(void*) method
		std::vector<state*> open_list; //A* open list: state node elements sorted by path cost
		std::vector<state*> close_list; //A* close list: state node elements that have been processed
		std::vector<state*> registers; /*This vector is identical to the open list; however, the elements are sorted by register value.
									     The sole function of the registers vector is to allow a binary search by register value for state nodes currently on the open list.
										 The register_found member function implements the binary search. If a state node on the open list has a register node identical to
										 the state& parameter then a pointer to the state on the open list is returned. If a state node on the open list does NOT have 
										 an identical register value nullptr is returned.
									   */
		//Implements binary search of register values on open_list. Returns pointer to state on open_list if match is found, otherwise nullptr is returned.
		state* register_found(const state&);

		/*Moves state& argument to address of state* argument if state*->pathcost is GT state&.pathcost and returns true, otherwise takes no action and returns false.
		Designed for use in conjunction with register_found(const state&) {e.g. bool update_state(.
		WARNING: state* must point to a state object! Ensure state* argument is not null before calling update state.*/
		bool update_state(state*, const state&);

		//Inserts state* into registers
		void insert_by_register(state*);
		//Inserts state* into open_list
		void insert_by_path_cost(state*);
		//resorts open_list
		void resort_by_path_cost();
		//resorts registers
		void resort_by_register_value();

		//prints content of open_list
		void print_open_list_function(std::ostream&) const;
		//prints content of registers
		void print_registers_function(std::ostream&) const;
		//prints content of closed_list
		void print_closed_list_function(std::ostream&) const;

	public:
		//constructors
		state_list() {}
		~state_list();

		//if state.register_value is not in current open_list then added state to current list, otherwise, if state.register_value is in open_list and state.path_cost is less than open_list's state.path_cost then move content of state parameter to address of open_list state pointer
		void enqueue(const state&);
		//returns top of open_list, moves top of open_list to end of closed_list, erases open_list.begin()
		state& top();

		//public print methods
		inline void PrintRegisters() const { print_registers_function(std::cout); }
		inline void PrintOpenList() const { print_open_list_function(std::cout); }
		inline void PrintClosedList() const { print_closed_list_function(std::cout); }

		//friends
		friend std::ostream& operator<<(std::ostream& s, const state_list& a) {
			a.print_open_list_function(s);
			return s;
		}
	};
}
#endif