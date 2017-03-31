//CS4793 - progex1 - prob #1 - A* implementation
//Written By: Justin Lye

#include"state_list.h"

//returns a pointer to the state object if found and a nullptr otherwise;
jel::state* jel::state_list::register_found(const state& target) {
	jel::state* returned_ptr = nullptr;
	int strt_i = 0;
	int end_i = registers.size();
	int pvt_i = 0;
	while (strt_i < end_i) {
		pvt_i = floor((strt_i + end_i) / 2);
		if (target.RegisterValue() == registers[pvt_i]->RegisterValue()) {
			returned_ptr = registers[pvt_i];
			break;
		} else if (target.RegisterValue() < registers[pvt_i]->RegisterValue()) {
			end_i = pvt_i - 1;
		} else {
			strt_i = pvt_i + 1;
		}
	}
	if ((end_i == strt_i) && returned_ptr == nullptr && strt_i < registers.size()) {
		if (target.RegisterValue() == registers[strt_i]->RegisterValue()) {
			returned_ptr = registers[strt_i];
		}
	}
	return returned_ptr;
}

bool jel::state_list::update_state(state* state_ptr, const state& target) {
	bool result = false;
	if (state_ptr != nullptr) {
		if (state_ptr->PathCost() > target.PathCost()) {
			*state_ptr = target;
			result = true;
		}
	}
	return result;
}

//this should only be called after the register was not found
void jel::state_list::insert_by_register(state* state_ptr) {
	int offset = -1;
	for (int i = 0; i < registers.size(); i++) {
		if (state_ptr->RegisterValue() < registers[i]->RegisterValue()) {
			offset = i;
			break;
		}
	}

	if (offset == -1) {
		registers.push_back(state_ptr);
	} else {
		std::vector<jel::state*>::iterator iter = (registers.begin() + offset);
		registers.insert(iter, state_ptr);
	}

}

void jel::state_list::insert_by_path_cost(state* state_ptr) {
	int offset = -1;
	for (int i = 0; i < open_list.size(); i++) {
		if (state_ptr->PathCost() < open_list[i]->PathCost()) {
			offset = i;
			break;
		}
	}

	if (offset == -1) {
		open_list.push_back(state_ptr);
	} else {
		std::vector<jel::state*>::iterator iter = (open_list.begin() + offset);
		open_list.insert(iter, state_ptr);
	}
}

void jel::state_list::resort_by_path_cost() {
	std::sort<std::vector<jel::state*>::iterator>(open_list.begin(), open_list.end(), jel::state::pathcomp_LT);
}

void jel::state_list::resort_by_register_value() {
	std::sort<std::vector<jel::state*>::iterator>(registers.begin(), registers.end(), jel::state::keycomp_LT);
}

void jel::state_list::print_open_list_function(std::ostream& s) const {
	for (int i = 0; i < open_list.size()-1; i++) {
		s << *open_list[i] << ' ';
	}
	s << *open_list[open_list.size()-1] << '\n';
}

void jel::state_list::print_registers_function(std::ostream& s) const {
	for (int i = 0; i < registers.size() - 1; i++) {
		s << *registers[i] << ' ';
	}
	s << *registers[registers.size() - 1] << '\n';
}

void jel::state_list::print_closed_list_function(std::ostream& s) const {
	for (int i = 0; i < close_list.size() - 1; i++) {
		s << *close_list[i] << ' ';
	}
	s << *close_list[close_list.size() - 1] << '\n';
}

jel::state_list::~state_list() {
	for (int i = 0; i < open_list.size(); i++) {
		if (open_list[i] != nullptr) {
			delete open_list[i];
			open_list[i] = nullptr;
		}
		
	}
}

void jel::state_list::enqueue(const jel::state& s) {

	if(update_state(register_found(s), s)) {
		resort_by_path_cost();
	} else {
		//jel::state* state_ptr = nullptr;
		jel::state* state_ptr = new jel::state(s.RegisterValue(), s.Hcost(), s.EdgeCost(),s.ParentOp(),s.parent_state);
		//*state_ptr = s;
		insert_by_path_cost(state_ptr);
		insert_by_register(state_ptr);
		//trash_bin.insert(state_ptr);
	}
}

jel::state& jel::state_list::top() {
	jel::state* _top = nullptr;
	if (open_list.size() <= 0) {
		throw std::runtime_error("value not found");
	}
	resort_by_path_cost();
	_top = open_list[0];
	open_list.erase(open_list.begin());
	close_list.push_back(_top);
	return *_top;


}