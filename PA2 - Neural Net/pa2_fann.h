#ifndef __PA2_FANN_H__
#define __PA2_FANN_H__

/* declaration file for pa2_fann ....a feedforward neural network.
 *
 * Use as a baseline comparison for the sparse autoencoding pre-training of PA2.
 *
 * Feel free to use as needed for C++ implementation of PA2.
 */

// Doug Heisterkamp
// last modified: 10/29/16

#include<memory>
#include<Eigen/Core>
#include<utility>

namespace CS4793
{

   // group per node storage components into a structure
   struct Nodes {
      Eigen::MatrixXd  output;  // f(network activation) or input values  
      // additional per node storage for work 
      Eigen::MatrixXd  delta;  // sensititives for backpropagation
      Eigen::MatrixXd  error;  // error terms for backpropagation
      Eigen::MatrixXd  fprime;  // first derivative = f'(network activation)
      Eigen::MatrixXd  network;   // network activation = dot(w,x)  
      Nodes(int rows, int cols);  // construct all objects to have the same size
   };

   // group per weight storage components into a structure
   struct Links {  // between layer weights and bias
         Eigen::MatrixXd  W;  // weights  
         Eigen::RowVectorXd  b;  // bias 
         Eigen::MatrixXd  DeltaW;  // backprogation storage   
         Eigen::RowVectorXd  DeltaB;  // backprogation storage   
         Links(int rows, int cols);  // construct all objects to have the compatible sizes
   };


   // helper structure to name components returned from evaluating the ANN
   struct ErrorMeasures {
      double meanSquareError;
      double classificationErrorRate;
      double meanAbsLayer1FPrime; // add for viewing the saturation of the gradient
      double meanAbsLayer2FPrime; // add for viewing the saturation of the gradient
      double meanAbsLayer3FPrime; // add for viewing the saturation of the gradient
   };


   class PA2_FANN
   {
      public:
         // trying to use similar names as the python tensorflow implementation
         static const int batchsize; // mini-batch training
         static const int maxEpochs; // mini-batch training
         static const double lr_init; // learning rate inital value
         static const bool update_lr;   // change learning rate during trainig?
         static const bool useReg;     // use weight regularization
         static const double beta;     // scaling of weight regularization
		 int max_epochs;
         PA2_FANN(int inputDim, int outputDim,int hidden1, int hidden2);

         void initForTraining();
         int train(Eigen::MatrixXd &data, Eigen::MatrixXd &targets, Eigen::MatrixXd &valData, Eigen::MatrixXd &valTagets);
         void predict(Eigen::MatrixXd &indata, Eigen::MatrixXd &outTargets );
         ErrorMeasures evaluate(Eigen::MatrixXd &data, Eigen::MatrixXd &targets );

         void clearTrainingWork();
         void forwardPass(int sampleSize);
         void setErrors(int sampleSize);
         void backwardPass(int sampleSize);
		 void preTrainBackwardPass(int sampleSize);
         void updateWeights(int sampleSize);
		 void klDiv(int sampleSize);
         void altBackwardPass(int sampleSize);

         // data members
         int numFeatures; // 16 for letter data
         int numClasses;  // 26 for letter data
         int layer1;      // 70 for PA2
         int layer2;      // 50 for PA2
		 double srate;		// sparsity rate for PA2
		 double kl_beta;   //kl-div beta for PA2
		 bool stage1;
		 bool stage2;
		 bool stage3;

         
         // trying to use similar names as the python tensorflow implementation
         Nodes  xInput;   // hold a mini-batch of samples, row order; reusing Nodes which waste some storage
         Eigen::MatrixXd  yTargets;  // hold 1-hot target encoding of samples

         // parameters of FANN
         Links  W1a;  // weights from inputs to first hidden layer 
         Links  W2a;  // weights from first hidden layer to second hidden layer 
         Links  W3;  // weights from first hidden layer to second hidden layer 

         // computed components = f(network activations)
         Nodes  h1;  // output of hidden layer 1  
         Nodes  h2;  // output of hidden layer 3  
         Nodes  yPred;  // output layer  values 

         double lrate;
   };

}

#endif
