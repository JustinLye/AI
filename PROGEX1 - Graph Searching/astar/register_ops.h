//CS4793 - progex1 - prob #1 - A* implementation
//Written By: Justin Lye

#if !defined(__JEL_REGISTER_OPS_HEADER__)
#define __JEL_REGISTER_OPS_HEADER__
#include<iostream>
namespace jel {
#if !defined(__JEL_REGISTER_OPS__)
#define __JEL_REGISTER_OPS__
	//enumeration of possible register operations
	enum op { NONE, ONE, ADD, SUB, DUB, DIV };
#endif
	class operation {
	public:
		op oper;
		operation(op o) : oper(o) {}
		operator int() {
			switch (this->oper) {
			case NONE:
				return 0;
				break;
			case ONE:
				return 1;
				break;
			case ADD:
				return 2;
				break;
			case SUB:
				return 3;
				break;
			case DUB:
				return 4;
				break;
			case DIV:
				return 5;
				break;
			default:
				return -1;
				break;
			}
		}
		friend std::ostream& operator<<(std::ostream& s, jel::operation o) {
			switch (o.oper) {
			case NONE:
				s << "NO OP";
				break;
			case ONE:
				s << "ONE";
				break;
			case ADD:
				s << "ADD";
				break;
			case SUB:
				s << "SUB";
				break;
			case DUB:
				s << "DOUBLE";
				break;
			case DIV:
				s << "DIVIDE";
				break;
			}
			return s;
		}
	};


}



#endif