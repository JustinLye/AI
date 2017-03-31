#include"nn_structs.h"

using namespace nn;

void nodes::resize(int rows, int cols) {
	network.resize(rows, cols);
	output.resize(rows, cols);
	target.resize(rows, cols);
	prime.resize(rows, cols);
	error.resize(rows, cols);
	delta.resize(rows, cols);
	network.setZero();
	output.setZero();
	target.setZero();
	prime.setZero();
	error.setZero();
	delta.setZero();
}

void nodes::clear_training_work() {
	error.setZero();
	delta.setZero();
}
void links::resize(int rows, int cols) {
	weights.resize(rows, cols);
	bias.resize(cols);
	weights_delta.resize(rows, cols);
	bias_delta.resize(cols);
	weights.setZero();
	bias.setZero();
	weights_delta.setZero();
	bias_delta.setZero();
}

void links::clear_training_work() {
	weights_delta.setZero();
	bias_delta.setZero();
}

void layer::resize(int rows, int indims, int hiddendims) {
	hidden_dims = hiddendims;
	input.resize(rows,indims);
	hidden.resize(rows, hidden_dims);
	output.resize(rows, indims);
	inlink.resize(indims, hidden_dims);
	outlink.resize(hidden_dims, indims);
}

void layer::randomize_links() {
	inlink.weights.setRandom() *= (1.0 / (1.0 + sqrt(inlink.weights.cols())));
	inlink.bias.setRandom() *= (1.0 / (1.0 + sqrt(inlink.weights.cols())));
	outlink.weights.setRandom() *= (1.0 / (1.0 + sqrt(outlink.weights.cols())));
	outlink.bias.setRandom() *= (1.0 / (1.0 + sqrt(outlink.weights.cols())));
}

void layer::clear_training_work() {
	input.clear_training_work();
	output.clear_training_work();
	hidden.clear_training_work();
	inlink.clear_training_work();
	outlink.clear_training_work();
}

void encoder::initialize(const policy& p, const char* filename) {
	hidden_dims = p.hidden_dims;
	batchsize = p.batch_size;
	max_epoch = p.max_epoch;
	init_lrate = p.init_lr;
	lrate = p.init_lr;
	lr_interval = p.lr_update_interval;
	generate_net(filename);
	set_indices();
	noise_input();
	srate = 1.0 / double(hidden_dims);
	kl_beta = 0.00001;
	wg_beta = 0.1/(input_dims * hidden_dims + input_dims*hidden_dims + input_dims*hidden_dims);
	eval_interval = 10;
	net.inlink.name = "Input Links";
	net.input.name = "Input Nodes";
	net.outlink.name = "Output Links";
	net.output.name = "Output Nodes";
	net.hidden.name = "Hidden Nodes";
	

}

void encoder::generate_net(const char* filename) {
	mat raw_data;
	nn::read_raw_data(filename, raw_data);
	input_dims = raw_data.cols() - 1;
	rows = raw_data.rows();

	orig_data = raw_data.block(0,1,raw_data.rows(), input_dims);
	net.resize(rows, input_dims, hidden_dims);
	net.randomize_links();
	net.output.target = raw_data.block(0, 1, raw_data.rows(), input_dims);
	net.input.output = raw_data.block(0, 1, raw_data.rows(), input_dims);
	noisy_data.resize(rows, input_dims);
}

void encoder::set_indices() {
	sample_indices.clear();
	for (int i = 0; i < rows; i++) {
		sample_indices.push_back(i);
	}
}

void encoder::noise_input() {
	noise_maker.AddGaussianNoise(orig_data, noisy_data);
}

void encoder::clear_training_work() {
	net.clear_training_work();
}

void encoder::shuffle() {
	noise_input();
	std::random_shuffle(sample_indices.begin(), sample_indices.end());
	for (int i = 0; i < rows; i++) {
		net.input.output.row(i) = noisy_data.row(sample_indices[i]);
		net.output.target.row(i) = orig_data.row(sample_indices[i]);
	}
}

void encoder::set_for_epoch() {
	base_index = 0;
	samples = 0;
	clear_training_work();
	shuffle();
}

void encoder::episode() {
	set_for_epoch();
	batch();
	if ((epoch + 1) % lr_interval == 0) {
		lrate = std::max(init_lrate / (1.0 + sqrt(epoch)), 0.00001);
		lr_interval *= 2;
	}
	epoch++;
}

void encoder::batch() {
	int bpe = int(std::ceil(rows/double(batchsize)));
	for (int i = 0; i < bpe; i++) {
		clear_training_work();
		base_index = i*batchsize;
		this->samples = std::min(batchsize, rows-base_index);
		mini_batch();
	}
}

void encoder::mini_batch() {
	feed_forward();
	set_error();
	backpropogate();
	update_weights();
}

void encoder::feed_forward() {
	net.hidden.network.block(base_index, 0, samples, hidden_dims) =
		(net.input.output.block(base_index, 0, samples, input_dims) * net.inlink.weights).rowwise() + net.inlink.bias;
	net.hidden.output.block(base_index, 0, samples, hidden_dims) =
		1.0 / (Eigen::exp(net.hidden.network.block(base_index, 0, samples, hidden_dims).array() * -1.0) + 1.0);
	net.hidden.prime =
		net.hidden.output.block(base_index, 0, samples, hidden_dims).array() * (1.0 - net.hidden.output.block(base_index, 0, samples, hidden_dims).array());

	net.output.network.block(base_index, 0, samples, input_dims) =
		(net.hidden.output.block(base_index, 0, samples, hidden_dims) * net.outlink.weights).rowwise() + net.outlink.bias;
	net.output.output.block(base_index, 0, samples, input_dims) =
		1.0 / (Eigen::exp(net.output.network.block(base_index, 0, samples, input_dims).array() * -1.0) + 1.0);
	net.output.prime =
		net.output.output.block(base_index, 0, samples, input_dims).array() * (1.0 - net.output.output.block(base_index, 0, samples, input_dims).array());
}

void encoder::set_error() {
	net.output.error.block(base_index, 0, samples, input_dims) = net.output.target.block(base_index, 0, samples, input_dims) - net.output.network.block(base_index, 0, samples, input_dims);
	set_penalty();
	net.outlink.weights_delta = net.outlink.weights * (-wg_beta);
	net.inlink.weights_delta = net.inlink.weights * (-wg_beta);
}

void encoder::set_penalty() {
	row_vec hmean = net.hidden.output.colwise().mean();
	row_vec dkldiv = -srate / hmean.array() + (1 - srate) / (1.0 - hmean.array());

	for (int i = 0; i < samples; i++) {
		net.hidden.error.row(base_index + i) = kl_beta * dkldiv.array();
	}
}

void encoder::backpropogate() {
	net.output.delta.block(base_index, 0, samples, input_dims) =
		net.output.error.block(base_index, 0, samples, input_dims).array() * net.output.prime.block(base_index, 0, samples, input_dims).array();
	net.outlink.weights_delta += lrate * (net.hidden.output.block(base_index, 0, samples, hidden_dims).transpose() * net.output.delta.block(base_index, 0, samples, input_dims) / (input_dims * samples));
	net.outlink.bias_delta += lrate * net.output.delta.block(base_index, 0, samples, input_dims).colwise().sum() / (samples * input_dims);

	net.hidden.delta.block(base_index, 0, samples, hidden_dims) =
		net.hidden.error.block(base_index, 0, samples, hidden_dims).array() + (net.output.delta.block(base_index, 0, samples, input_dims) * net.outlink.weights.transpose()).array()
		* net.hidden.prime.array();
	net.inlink.weights_delta += lrate * ((net.input.output.block(base_index, 0, samples, input_dims).transpose() * net.hidden.delta.block(base_index, 0, samples, hidden_dims)) / (hidden_dims * samples));

	net.inlink.bias_delta += lrate * (net.hidden.delta.block(base_index, 0, samples, hidden_dims).colwise().sum() / (hidden_dims * samples));
}

void encoder::update_weights() {
	net.outlink.weights += net.outlink.weights_delta;
	net.outlink.bias += net.outlink.bias_delta;
	net.inlink.weights += net.inlink.weights_delta;
	net.inlink.bias += net.inlink.bias_delta;
}

void encoder::train(bool print_epoch) {
	epoch = 0;
	lrate = init_lrate;
	while (epoch < max_epoch) {
		episode();
	}
	
}

double encoder::evaluate() {
	base_index = 0;
	samples = 0;
	int bpe = int(std::ceil(rows/double(batchsize)));
	double mse = 0.0;
	for (int i = 0; i < bpe; i++) {
		base_index = i * batchsize;
		samples = std::min(batchsize, rows - base_index);
		feed_forward();
		mse += (net.output.target.block(base_index, 0, samples, input_dims) - net.output.network.block(base_index, 0, samples, input_dims)).array().square().mean() * (samples/batchsize);
	}
	mse /= bpe;
	return mse;
}
