/* driver program for pa2_fann .... reads letter data and trains a feedforward neural network.
 *
 * Use as a baseline comparison for the sparse autoencoding pre-training of PA2.
 *
 * Feel free to use as needed for C++ implementation of PA2.
 */

// Doug Heisterkamp
// last modified: 10/28/16

#include<iostream>
#include<fstream>
#include<random>
#include<ctime>

#include"pa_util.h"
#include"pa2_fann.h"
#include"nn_structs.h"

using std::endl;
using std::cout;
using std::cerr;
int main(int argc, char *argv[]){
   const double inTarget = 0.9;  // value to use to true class in 1-hot code
   const double outTarget = 0.1; // value to use for wrong classes in 1-hot code
   srand(time(NULL));

   // read data files ... change path as needed
   std::ifstream trainFile("letterTrain.txt");
   if (! trainFile) {
      cout << "Failed to open letterTrain.txt data file.   Exiting." << endl;
      exit(-1);
   }

   std::ifstream valFile("letterValidation.txt");
   if (! valFile) {
      cout << "Failed to open letterValidation.txt data file.   Exiting." << endl;
      exit(-1);
   }

   // training data storage
   Eigen::MatrixXd trainData;
   Eigen::MatrixXd train1Hot;
   std::vector<int> trainTags;
   std::set<int>  trainUTags;
   std::map<int,int> trainTagDict;
   // read training data  
   if (! CS4793::loadDataFile( trainFile, trainData, train1Hot,inTarget,outTarget,trainTags,trainUTags,trainTagDict) ) {
      cout << "Failed to properly load letter training data.  Exiting." << endl;
      exit(-2);
   }

   // validation data storage
   Eigen::MatrixXd valData;
   Eigen::MatrixXd val1Hot;
   std::vector<int> valTags;
   std::set<int>  valUTags;
   std::map<int,int> valTagDict;
   // read valing data  
   if (! CS4793::loadDataFile( valFile, valData,val1Hot,inTarget,outTarget,valTags,valUTags,valTagDict) ) {
      cout << "Failed to properly load letter validation data.  Exiting." << endl;
      exit(-2);
   }


   CS4793::PA2_FANN fann(16,26,70,50);
   
   std::cout << "Pre-train? (y/n): ";
   char pretrain = 'n';
   std::cin >> pretrain;
   if (pretrain == 'y') {
	   std::cout << "pretraining..." << std::endl;
	   //stage 1:
	   nn::policy stage1_policy;
	   //weights delta converged to -nan when using mini-batch training
	   stage1_policy.batch_size = 12000;
	   stage1_policy.hidden_dims = 70;
	   stage1_policy.init_lr = 1;
	   stage1_policy.lr_update_interval = 5;
	   stage1_policy.max_epoch = 50;
	   nn::encoder stage1;
	   stage1.initialize(stage1_policy, "letterTrain.txt");
	   stage1.train();
	   std::ofstream stage2_input;
	   stage2_input.open("stage2_input.txt");
	   Eigen::MatrixXd input_copy;
	   input_copy.setOnes(stage1.net.hidden.network.rows(), stage1.net.hidden.network.cols()+1);
	   input_copy.block(0,1,input_copy.rows(), input_copy.cols()-1) = stage1.net.hidden.network.block(0,0,stage1.net.hidden.network.rows(), stage1.net.hidden.network.cols());
	   stage2_input << input_copy;
	   stage2_input.close();
	   //stage 2:

	   nn::policy stage2_policy;
	   stage2_policy.batch_size = 12000;
	   stage2_policy.hidden_dims = 50;
	   stage2_policy.init_lr = 1;
	   stage2_policy.lr_update_interval = 5;
	   stage2_policy.max_epoch = 50;
	   nn::encoder stage2;
	   stage2.initialize(stage2_policy, "stage2_input.txt");
	   stage2.train();
	   fann.initForTraining();
	   fann.W1a.b = stage1.net.inlink.bias;
	   fann.W1a.W = stage1.net.inlink.weights;
	   fann.W2a.W = stage2.net.inlink.weights;
	   fann.W2a.b = stage2.net.inlink.bias;
	   fann.max_epochs = 400;
   } else {
	   fann.initForTraining();
   }

   int num_epochs = fann.train(trainData,train1Hot,valData,val1Hot);
   cout << "After " << num_epochs << " of training : " << endl;
   auto trerror = fann.evaluate(trainData,train1Hot);
   cout << "   training mean square error           = " << trerror.meanSquareError << endl;
   cout << "   training classification error rate   = " << trerror.classificationErrorRate << endl;
   auto valerror = fann.evaluate(valData,val1Hot);
   cout << "   validation mean square error         = " << valerror.meanSquareError << endl;
   cout << "   validation classification error rate = " << valerror.classificationErrorRate << endl;

   cout << "Eigen used "<< Eigen::nbThreads( ) << "  threads " << endl;

   return 0;
}
