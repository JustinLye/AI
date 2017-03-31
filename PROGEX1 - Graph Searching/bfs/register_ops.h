
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