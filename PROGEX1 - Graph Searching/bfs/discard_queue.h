#if !defined(__JEL_DISCARD_QUEUE_HEADER__)
#define __JEL_DISCARD_QUEUE_HEADER__
/*
	the trash class is a supplement to a queue. whenever the queue pops, instead of immediately
	releasing memory, the queue will insert the address of the popped object into this queue.
	when this queue is destroyed it will also destroy the objects inserted into the queue.

    this class is an attempt to make sure allocated memory is released with the
	state_queue is destroyed. It seems to work...using the vs2015 memory allocation tracking tool
	it appears all memory allocated is released at the time main returns.

	using this class may avoid memory leak; however it is very inefficent.
*/

namespace jel {
	class trash {
	public:
		void* house_number;
		trash* next;
		trash() : house_number(nullptr), next(nullptr) {}
		trash(void* location) : house_number(location), next(nullptr) {}
	};

	class discard_queue {
	public:
		trash* trail_head;
		discard_queue() {
			trail_head = new trash();
			trail_head->next = nullptr;
		}
		~discard_queue() {
			trash* temp = nullptr;
			int count = 0;
			std::cout << "\n\ndiscard_queue start\n\n";
			while (trail_head != nullptr) {
				temp = trail_head;
				trail_head = trail_head->next;
				if (temp->house_number != nullptr) {
					std::cout << "\nfreeing house " << count << "\n";
					free(temp->house_number);
					std::cout << "\nhouse free\n";
				}
				std::cout << "\nfreeing trash " << count << "\n";
				free(temp);
				std::cout << "\ntrash free\n";
				count++;
			}
			std::cout << "\n\ndiscard_queue end\n\n";
		}
		void insert(void* location) {
			trash* temp = trail_head;
			trail_head = new trash(location);
			trail_head->next = temp;
		}
	};
}
#endif