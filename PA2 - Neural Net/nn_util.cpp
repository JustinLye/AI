#include "nn_util.h"

void nn::process_raw_data(
	mat& raw_data,
	mat& input_data,
	mat& target_data,
	double true_target,
	double wrong_target) {

	int num_rows = raw_data.rows();
	int num_cols = raw_data.cols() - 1;
	mat tags;
	std::set<int> unique_tags;
	std::map<int,int> tag_to_node;

	input_data.resize(num_rows, num_cols);
	input_data.block(0,0,num_rows,num_cols) = raw_data.block(0, 1, num_rows, num_cols);

	tags.resize(num_rows, 1);
	tags.block(0,0,num_rows,1) = raw_data.block(0, 0, num_rows, 1);

	for (int r = 0; r < num_rows; r++)
		unique_tags.insert(int(tags(r, 0)));

	int num_classes = int(unique_tags.size());
	
	for (int t : unique_tags)
		tag_to_node[t] = int(tag_to_node.size());

	assert(tag_to_node.size() == unique_tags.size());

	target_data.resize(num_rows, num_classes);
	target_data.fill(wrong_target);

	for (int r = 0; r < num_rows; r++) {
		auto p = tag_to_node.find(tags(r, 0));
		assert(p != tag_to_node.end());
		target_data(r, (p->second - 1)) = true_target;
	}

}

bool nn::process_raw_data(
	std::istream& in,
	mat& input_data,
	mat& target_data,
	double true_target,
	double wrong_target) {

	return CS4793::loadDataFile(in, input_data, target_data, true_target, wrong_target);
}

bool nn::read_raw_data(std::istream& in, mat& raw_data) {
	std::vector<std::vector<double> > input_vector;
	if (!CS4793::readDataAsVecOfVecs(in, input_vector)) {
		return false;
	}
	copy_from_stdvector(input_vector, raw_data);
	return true;

}

bool nn::read_raw_data(const char* filename, mat& raw_data) {
	bool result = false;
	std::ifstream in;

	in.open(filename);
	result = read_raw_data(in, raw_data);
	in.close();
	return result;

}

void nn::copy_from_stdvector(std::vector<std::vector<double> >& data_vector, mat& data_matrix) {
	int num_rows = int(data_vector.size());
	int num_cols = int(data_vector[0].size());

	data_matrix.resize(num_rows, num_cols);
	for (int row = 0; row < num_rows; ++row) {
		for (int col = 0; col < num_cols; ++col) {
			data_matrix(row,col) = data_vector[row][col];
		}
	}
}
