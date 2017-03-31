#if !defined(__JEL_GRAPH_HEADER__)
#define __JEL_GRAPH_HEADER__
#include<ctime>
#include"state_queue.h"

namespace jel {

	class graph {
	private:
		double runtime;
		long _expanded_nodes;
		long _queue_size;
		inline void Expand() {
			_expanded_nodes++;
			_queue_size--;
			jel::state* current_node = &_queue.pop();
			if (current_node->register_value % 3 == 0) {
				_queue.insert(current_node->register_value / 3, current_node->path_cost + (2 / 3)*current_node->register_value, jel::DIV, current_node);
				_queue_size++;
			} else {
				_queue.insert(current_node->register_value + 1, current_node->path_cost + 1, jel::ADD, current_node);
				_queue_size++;
				if (current_node->register_value >= 1) {
					_queue.insert(current_node->register_value * 2, current_node->path_cost + current_node->register_value, jel::DUB, current_node);
					_queue.insert(current_node->register_value - 1, current_node->path_cost + 1, jel::SUB, current_node);
					
					_queue_size += 2;
				}
			}
		}

	public:
		jel::state_queue _queue;
		graph() : _expanded_nodes(0), _queue_size(1) {
			_queue.insert(1,1,jel::ONE,nullptr);
		}


		long NodesExpanded() const {
			return this->_expanded_nodes;
		}
		long QueueSize() const {
			return this->_queue_size;
		}
		inline double Runtime() const { return runtime; }
		jel::state& FindTest(int target) {
			std::clock_t begin = clock();
			while (target != _queue.peek()->register_value) {
				this->Expand();
			}
			std::clock_t end = clock();
			runtime = double(end-begin)/CLOCKS_PER_SEC;
			return _queue.pop();
		}

		jel::state& Find(int target) {
			if (target == _queue.peek()->register_value) {
				return _queue.pop();
			} else {
				Expand();
				this->Find(target);
			}
		}


	};
}

#endif
