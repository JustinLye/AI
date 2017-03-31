#if !defined(__NN_STRUCTS_HEADER__)
#define __NN_STRUCTS_HEADER__
#include "nn_util.h"
#include "noise.h"
namespace nn {

	class nodes {
	public:
		mat network;
		mat output;
		mat target;
		mat prime;
		mat error;
		mat delta;
		const char* name;
		void resize(int rows, int cols);
		void clear_training_work();
		friend std::ostream& operator<<(std::ostream& s, const nodes& n) {
			s << n.name << std::endl << std::endl <<
				n.name << " Network" << std::endl << n.network  << std::endl << std::endl <<
				n.name << " Output" << std::endl << n.output << std::endl << std::endl <<
				n.name << " Target" << std::endl << n.target << std::endl << std::endl <<
				n.name << " Prime" << std::endl << n.prime << std::endl << std::endl <<
				n.name << " Error" << std::endl << n.error << std::endl << std::endl <<
				n.name << " Delta" << std::endl << n.delta << std::endl << std::endl;
			return s;
		}
	};

	class links {
	public:
		mat weights;
		row_vec bias;
		mat weights_delta;
		row_vec bias_delta;
		const char* name;
		void resize(int rows, int cols);
		void clear_training_work();
		friend std::ostream& operator<<(std::ostream& s, const links& l) {
			s << l.name << std::endl << std::endl <<
				l.name << " Weights" << std::endl << l.weights << std::endl << std::endl <<
				l.name << " Bias" << std::endl << l.bias << std::endl << std::endl <<
				l.name << " Weights Delta" << std::endl << l.weights_delta << std::endl << std::endl <<
				l.name << " Bias Delta" << std::endl << l.bias_delta << std::endl << std::endl;
			return s;
		}
	};

	class layer {
	public:
		nodes input;
		nodes hidden;
		nodes output;
		links inlink;
		links outlink;
		int hidden_dims;
		void resize(int rows,int indims, int hiddendims);
		void clear_training_work();
		void randomize_links();
		friend std::ostream& operator<<(std::ostream& s, const layer& l) {
			s << l.input << l.inlink << l.hidden << l.outlink << l.output;
			return s;
		}
	};

	struct policy {
		int hidden_dims;
		double init_lr;
		int lr_update_interval;
		int batch_size;
		int max_epoch;
		int input_start_col;
	};

	class encoder {
	public:
		layer net;
		mat orig_data;
		mat noisy_data;
		double srate;
		double init_lrate;
		double lrate;
		int batchsize;
		int epoch;
		int max_epoch;
		double kl_beta;
		double wg_beta;
		int rows;
		int input_dims;
		int hidden_dims;
		int base_index;
		int samples;
		int eval_interval;
		int lr_interval;
		std::vector<int> sample_indices;
		noise noise_maker;

		void initialize(const policy& p, const char* filename);
		void generate_net(const char* filename);
		void set_indices();
		void noise_input();
		void shuffle();
		void set_for_epoch();
		void clear_training_work();
		void episode();
		void batch();
		void mini_batch();
		void feed_forward();
		void set_error();
		void backpropogate();
		void set_penalty();
		void update_weights();
		double evaluate();
		void train(bool print_epoch = false);
		std::string output_filename;
		int print_interval;
		std::ofstream output_file;
		void feed_forward_alt();

		friend std::ostream& operator<<(std::ostream& s, const encoder& e) {
			s << "Sparsity Rate: " << e.srate << std::endl <<
				"Init LR: " << e.init_lrate << std::endl <<
				"LR: " << e.lrate << std::endl <<
				"BS: " << e.batchsize << std::endl <<
				"Epoch: " << e.epoch << std::endl <<
				"Max Epoch: " << e.max_epoch << std::endl <<
				"KL beta: " << e.kl_beta << std::endl <<
				"WG beta: " << e.wg_beta << std::endl << std::endl;
			return s;
		}
	};
};

#endif