#include<Eigen/Core>
#include<iostream>
#include<random>
#include<vector>
#include<ctime>
#include<algorithm>


#if !defined(__AI_NOISE_HEADER__)
#define __AI_NOISE_HEADER__

namespace nn {
	class noise_policy {
	public:
		static const double MIN_SWAPS;
		static const double MAX_SWAPS;
		static const int MAX_REDRAWS;
	private:
		noise_policy() {}
		noise_policy(const noise_policy& n) {}
		noise_policy(noise_policy&& n) {}
	};

	class noise {
	public:
		noise();
		void RandomZeroOut(std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& output, double zero_out_pct = 0.40);
		void RandomZeroOut(Eigen::MatrixXd& input, Eigen::Ref<Eigen::MatrixXd> output, double zero_out_pct = 0.40);
		void RandomZeroOut(Eigen::Ref<Eigen::MatrixXd> input, double zero_out_pct = 0.40);
		void AddGaussianNoise(std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& output, double std_dev = 0.001);
		void AddGaussianNoise(Eigen::MatrixXd& input, Eigen::Ref<Eigen::MatrixXd> output, double std_dev = 0.001);
		void AddGaussianNoise(Eigen::Ref<Eigen::MatrixXd> input, double std_dev = 0.001);
	protected:
		std::random_device rd;
		std::mt19937 gen;
		std::uniform_int_distribution<> dist;
		std::normal_distribution<double> norm_dist;
		void ZeroOut(std::vector<double>& output_vec, double zero_out_pct);
		void ZeroOut(Eigen::Ref<Eigen::MatrixXd> output,int col, double zero_out_pct);
		void GaussianNoise(std::vector<double>& output_vec);
		void GaussianNoise(Eigen::Ref<Eigen::MatrixXd> output, int col);
	};
}

#endif
