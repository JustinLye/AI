//20160922 - Justin Lye: Created File
#if !defined(__JEL_STATE_NODE_HEADER__)
#define __JEL_STATE_NODE_HEADER__

/* data structure to represent state nodes in graph search
   using operations as arcs and cost as n for op DUB
   (2/3) * n for DIV and 1 for others
*/

#include<iostream>
#include"register_ops.h"

namespace jel {
	class state {
	protected:
		//protected members
		int register_value;
		int h_cost;
		int edge_cost;
		operation parent_op;
		
		state* next; //for queue
		void print_function(std::ostream& s) const {
			s << parent_op << "--> " << register_value <<  " [" << edge_cost + h_cost << " = " << edge_cost << " + " << h_cost << "] ";
		}
		void _PrintHistory(state* s) {
			if (s == nullptr) {
				return;
			} else {
				_PrintHistory(s->parent_state);
				std::cout << *s;
			}
		}
	public:
		state* parent_state;
		//constructor
		state() : register_value(0),  h_cost(0), edge_cost(0), parent_op(NONE), next(nullptr), parent_state(nullptr) {}
		state(jel::state& s) :
			register_value(s.register_value),
			h_cost(s.h_cost),
			edge_cost(s.edge_cost),
			parent_op(s.parent_op),
			parent_state(s.parent_state) {}
		state(int register_val, int target_value, operation par_op, state* parent) :
			register_value(register_val),
			h_cost(0),
			edge_cost(0),
			parent_op(par_op),
			next(nullptr),
			parent_state(parent) {
			if (parent_state != nullptr) {
				switch (int(par_op)) {
				case jel::op::ADD:
				case jel::op::SUB:
				case jel::op::ONE:
					edge_cost = parent->edge_cost + 1;
					break;
				case jel::op::DUB:
					edge_cost = parent->edge_cost + parent_state->register_value;
					break;
				case jel::op::DIV:
					edge_cost = parent->edge_cost + (2 * (parent_state->register_value/3));
					break;
				}
			} else if (par_op == jel::op::ONE) {
				edge_cost = 1;
			}
			
				
			h_cost = labs(target_value - register_value);
		}
		state(int register_val, int init_h_cost, int init_edge_cost, operation par_op, state* parent) :
			register_value(register_val),
			h_cost(init_h_cost),
			edge_cost(init_edge_cost),
			parent_op(par_op),
			next(nullptr),
			parent_state(parent) {}
		
		//getters
		inline int RegisterValue() const { return register_value; }
		inline int PathCost() const { return h_cost + edge_cost; }
		inline int Hcost() const { return h_cost; }
		inline int EdgeCost() const { return edge_cost; }
		inline operation ParentOp() const { return parent_op; }
		inline state* Parent() { return parent_state; }
		
		inline const state* Next() const { return next; }
		//setters
		inline void SetRegisterValue(int val) { register_value = val; }
		inline void SetParentOp(operation& ops) { parent_op = ops; }
		inline void SetParent(state* paren) { parent_state = paren; }
		inline void SetNext(state* nxt) { next = nxt; }
		//functions
		void PrintHistory() {
			this->_PrintHistory(this);
		}

		//comparision operators (path_cost is evaluated)
		bool operator<(const state& s) { return ((this->h_cost + this->edge_cost) < (s.h_cost + s.edge_cost)); }
		bool operator<=(const state& s) { return ((this->h_cost + this->edge_cost) <= (s.h_cost + s.edge_cost)); }
		bool operator>(const state& s) { return ((this->h_cost + this->edge_cost) > (s.h_cost + s.edge_cost)); }
		bool operator>=(const state& s) { return ((this->h_cost + this->edge_cost) >= (s.h_cost + s.edge_cost)); }
		bool operator<(const state* s) { return ((s->h_cost + s->edge_cost) < (this->h_cost + this->edge_cost)); }
		bool operator<=(const state* s) { return ((s->h_cost + s->edge_cost) <= (this->h_cost + this->edge_cost)); }
		bool operator>(const state* s) { return ((s->h_cost + s->edge_cost) > (this->h_cost + this->edge_cost)); }
		bool operator>=(const state* s) { return ((s->h_cost + s->edge_cost) >= (this->h_cost + this->edge_cost)); }
		bool operator==(const state& s) { return (s.register_value == this->register_value); }
		bool operator==(const state* s) { return (s->register_value == this->register_value); }
		

		//friends
		friend std::ostream& operator<<(std::ostream& s, const state& st) {
			st.print_function(s);
			return s;
		}
		

		//used for sorting
		static struct {
			bool operator()(const state* a, const state* b) {
				return ((a->h_cost+a->edge_cost) < (b->h_cost+b->edge_cost));
			}
		} pathcomp_LT;

		static struct {
			bool operator()(const state& a, const state& b) {
				return (a.register_value < b.register_value);
			}
			bool operator()(const state* a, const state* b) {
				return (a->register_value < b->register_value);
			}
		} keycomp_LT;

		static struct {
			bool operator()(const state& a, const state& b) {
				return (a.register_value <= b.register_value);
			}
		} keycomp_LE;

		static struct {
			bool operator()(const state& a, const state& b) {
				return (a.register_value > b.register_value);
			}
		} keycomp_GT;

		static struct {
			bool operator()(const state& a, const state& b) {
				return (a.register_value == b.register_value);
			}
		} keycomp_EQ;
	};



	

}

#endif


