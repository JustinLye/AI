#if !defined(__JEL_STATE_QUEUE_HEADER__)
#define __JEL_STATE_QUEUE_HEADER__


#include "state.h"
#include "discard_queue.h"
namespace jel {
	class state_queue {
	private:
		state* _head;
		state* _tail;
		discard_queue _trash;
		void print_function(std::ostream& s) const {
			state* printer = _head;
			while (printer != nullptr) {
				s << *printer << ' ';
				printer = printer->next;
			}
		}
	protected:
		state* peek() { return _head; }
	public:
		//constructor
		state_queue() : _head(nullptr), _tail(nullptr) {}
		//destructor
		~state_queue() {
			std::cout << "\n\nstate_queue start\n\n";
			state* del_ptr = nullptr;
			while (_head != nullptr) {
				del_ptr = _head;
				_head = _head->next;
				delete del_ptr;
				del_ptr = nullptr;
			}
			std::cout << "\n\nstate_queue end\n\n";
		}

		//methods
		//state* peek() { return _head; }
		void insert(int register_val, int cost, operation par_op, state* parent) {
			state* new_node = new state(register_val, cost, par_op, parent);
			
			if (_head == nullptr) {
				_head = new_node;
			} else if (_tail == nullptr) {
				_tail = new_node;
				_head->next = _tail;
			} else {
				_tail->next = new_node;
				_tail = new_node;
			}

		}

		state& pop() {
			state* return_state = nullptr;
			if (_head != nullptr) {
				return_state = _head;
				_trash.insert(_head); //attempt to ensure new_node is deleted
				_head = _head->next;
			}
			return *return_state;
		}

		//friends
		friend class graph;
		friend std::ostream& operator<<(std::ostream& s, const state_queue& sq) {
			sq.print_function(s);
			return s;
		}
	};
}

#endif
