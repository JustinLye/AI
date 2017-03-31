#include<fstream>
#include "pa_util.h"

#if !defined(__NN_FANN_UTIL_HEADER__)
#define __NN_FANN_UTIL_HEADER__


namespace nn {

#define NN_PREPARED_INPUT_VALUES 0x0000
#define NN_RAW_INPUT_VALUES 0x0001

#define NN_ZERO_OUT_METHOD 0x2010
#define NN_GAUSSIAN_METHOD 0x2011


	typedef Eigen::MatrixXd mat;
	typedef Eigen::RowVectorXd row_vec;
	typedef Eigen::VectorXd col_vec;
	typedef Eigen::MatrixXd* mat_ptr;
	typedef Eigen::RowVectorXd* vec_ptr;

	void process_raw_data(
		mat& raw_data,
		mat& input_data,
		mat& target_data,
		double true_target = 0.9,
		double wrong_target = 0.1);
	bool process_raw_data(
		std::istream& in,
		mat& input_data,
		mat& target_data,
		double true_target = 0.9,
		double wrong_target = 0.1);

	bool read_raw_data(std::istream& in, mat& raw_data);
	bool read_raw_data(const char* filename, mat& raw_data);
	void copy_from_stdvector(std::vector<std::vector<double> >& data_vector, mat& data_matrix);
};

#endif