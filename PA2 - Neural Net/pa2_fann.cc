/* implementation file for pa2_fann ....a feedforward neural network.
 *
 * Use as a baseline comparison for the sparse autoencoding pre-training of PA2.
 *
 * Feel free to use as needed for C++ implementation of PA2.
 */

// Doug Heisterkamp
// last modified: 10/29/16


#include"pa2_fann.h"
#include<cmath>
#include<cassert>
#include<iostream>
#include<vector>

// class constant data members  --> show be in a policy object
const int CS4793::PA2_FANN::maxEpochs = 500; 
const int CS4793::PA2_FANN::batchsize = 120; // mini-batch training
const double CS4793::PA2_FANN::lr_init  = 10.0; // learning rate inital value -- note: dividing by batchsize in backwardPass
const bool CS4793::PA2_FANN::update_lr = true;   // change learning rate during trainig?
const bool CS4793::PA2_FANN::useReg = true;     // use weight regularization
const double CS4793::PA2_FANN::beta = 0.1 /(16*70.0 + 70*50 + 50*26);     // scaling of weight regularization

const bool useAltBackProp = false;     // use matrix version of backwardPass or alt version with for loops

CS4793::Links::Links(int rows, int cols):W(rows,cols),b(cols),DeltaW(rows,cols),DeltaB(cols) { }
CS4793::Nodes::Nodes(int rows, int cols):output(rows,cols),delta(rows,cols),error(rows,cols),fprime(rows,cols),network(rows,cols) { }

// a couple of functions that may help debugging ... check if all of the elements in a Link or Node is finite; exit if not 
void checkallFinite(CS4793::Links &a,const char *mes = "") {
   if (!a.W.allFinite()) { std::cout << "not finite in W of " << mes << std::endl; exit(-3);}
   if (!a.b.allFinite()) { std::cout << "non finite in b of " << mes << std::endl; exit(-3);}
   if (!a.DeltaW.allFinite()) { std::cout << "non finite in DeltaW of " << mes << std::endl; exit(-3);}
   if (!a.DeltaB.allFinite()) { std::cout << "non finite in DeltaB of " << mes << std::endl; exit(-3);}
}

void checkallFinite(CS4793::Nodes &a,const char *mes=""){
   if (!a.output.allFinite()) { std::cout << "non finite in output of " << mes << std::endl; exit(-3);}
   if (!a.delta.allFinite()) { std::cout << "non finite in delta of " << mes << std::endl; exit(-3);}
   if (!a.error.allFinite()) { std::cout << "non finite in error of " << mes << std::endl; exit(-3);}
   if (!a.fprime.allFinite()) { std::cout << "non finite in fprime of " << mes << std::endl; exit(-3);}
   if (!a.network.allFinite()) { std::cout << "non finite in network of " << mes << std::endl; exit(-3);}
}

// helper function to print out error measurements with respect to targets of prediction of ann on data
void printEval(std::ostream &out, CS4793::PA2_FANN &ann, Eigen::MatrixXd &data, Eigen::MatrixXd &targets, const char *mes="",
               bool printFPrime=false){
   auto res = ann.evaluate(data,targets);
   out <<  mes << " mse = " << res.meanSquareError << ", error rate= " << res.classificationErrorRate;  
   if (printFPrime) {
       out << ";  |f'| 1 = " << res.meanAbsLayer1FPrime << ", |f'| 2  = " << res.meanAbsLayer2FPrime   
           << ", |f'| 3  = " << res.meanAbsLayer3FPrime;
   }
   out << std::endl;
}

/* constructor : using a fixed sized feed forward network for letter data --- 
 * Create initial sizes for data members, but does not initialize values.
 */
CS4793::PA2_FANN::PA2_FANN(int inputDim, int outputDim,int hidden1, int hidden2):numFeatures(inputDim),
   numClasses(outputDim),
   layer1(hidden1),
   layer2(hidden2),
   xInput(CS4793::PA2_FANN::batchsize,inputDim),
   yTargets(CS4793::PA2_FANN::batchsize,outputDim),
   W1a(inputDim,hidden1),
   W2a(hidden1,hidden2),
   W3(hidden2,outputDim),
   h1(CS4793::PA2_FANN::batchsize,hidden1),
   h2(PA2_FANN::batchsize,hidden2),
   yPred(PA2_FANN::batchsize,outputDim),
	srate(1.0/hidden1)
{
   lrate = lr_init;
}


/* set all weight parameters of ANN to small initial random values. */
void CS4793::PA2_FANN::initForTraining()
{
   // random initialization of weights, assumes random number seed has been set in calling program
   // note: eigen3's random reals are in -1.0 to 1.0 range instead of 0.0 to 1.0
   W1a.W.setRandom() *= (1.0 / (1.0 + sqrt(numFeatures)));
   W1a.b.setRandom() *= (1.0 / (1.0 + sqrt(numFeatures)));
   W2a.W.setRandom() *= (1.0 / (1.0 + sqrt(layer1)));
   W2a.b.setRandom() *= (1.0 / (1.0 + sqrt(layer1)));
   W3.W.setRandom() *= (1.0 / (1.0 + sqrt(layer2)));
   W3.b.setRandom() *= (1.0 / (1.0 + sqrt(layer2)));
   max_epochs = maxEpochs;
}


/* Use data to train the ANN for a fixed number of epochs (MaxEpoch) in mini-batches of batchsize.
 *    -- data should containing training samples stored rowwise
 *    -- targets should be in 1-hot encoding, number of rows should match data
 *    -- valData and valTargets are not used to stop training in this code, just for evaluation.
 *
 *    returns number of epochs of training
 */
int CS4793::PA2_FANN::train(Eigen::MatrixXd &data, Eigen::MatrixXd &targets,Eigen::MatrixXd &valData, Eigen::MatrixXd &valTargets) 
{  
   lrate = lr_init; // learning rate

   std::vector<int> sampleIndices(data.rows());  // for random ordering of minibatch samples
   for (int j=0; j <data.rows(); ++j) {
      sampleIndices[j] = j;
   }

   int bpe = int(std::ceil(data.rows()/double(batchsize))); // number batches per epoch
   // run maxEpochs of training of data
   for(int i = 0; i < max_epochs; ++i){
      //std::cout << "epoch " << i <<  std::endl;
      // random shuffle of mini-batch
      std::random_shuffle(sampleIndices.begin(), sampleIndices.end());
      for (int j=0; j < bpe; ++j) { // for each mini-batch j
         clearTrainingWork(); // zero error and delta terms for use in this mini-batch
         // copy batch data to input  ... actually moving data instead of selecting/slicing to keep syntax simple
         int baseIndex =j*batchsize;  // offset for start of batch
         int amount =std::min(batchsize, int(data.rows())-baseIndex); // length of batch, handles end case
         for(int k=0; k < amount; ++k){
            xInput.output.row(k) = data.row(sampleIndices[k+baseIndex]);
            yTargets.row(k) = targets.row(sampleIndices[k+baseIndex]);
         }

         forwardPass(amount);
         setErrors(amount);
         if (useAltBackProp){
           altBackwardPass(amount);
         } else {
           backwardPass(amount);
         }	
         updateWeights(amount);
      }
      if (i%10 == 0) { // print out evaluation info every 10 epoch
         std::cout << "at epoch " << i << std::endl;
         printEval(std::cout, *this, data,targets,        "   training  : ", true);
         printEval(std::cout, *this, valData, valTargets, "   validation: ",false);
      }
      if (update_lr) {
         lrate =  lr_init/(1.0 + sqrt(i));
      }
   }
   return max_epochs;  // not stop early by validation testing
}

void CS4793::PA2_FANN::clearTrainingWork(){
   // note: output, network, and fprime as set in forward pass.  Don't need to be zero'd
   W1a.DeltaW.setZero();
   W1a.DeltaB.setZero();
   W2a.DeltaW.setZero();
   W2a.DeltaB.setZero();
   W3.DeltaW.setZero();
   W3.DeltaB.setZero();
   h1.error.setZero();
   h1.delta.setZero();
   h2.error.setZero();
   h2.delta.setZero();
   yPred.error.setZero();
   yPred.delta.setZero();
}

void CS4793::PA2_FANN::forwardPass(int sampleSize){
  // setting network, output, and fprime at the nodes of the ANN
  // sampleSize <= batchsize  to handle last batch which may not use full matrix 
  assert(sampleSize <= batchsize);

  // forward to first hidden layer
  h1.network.topRows(sampleSize) = (xInput.output.topRows(sampleSize)*W1a.W).rowwise() + W1a.b; 
  // apply logistic sigmoid to the newtork output.   ---  array() mean componentwise operation
  // so 1/(e^(-network)+1) is calculated at each element
  h1.output.topRows(sampleSize) = 1.0/(Eigen::exp(h1.network.topRows(sampleSize).array()*-1.0) + 1.0);  
  // set the first derivate while doing the forward pass 
  // --- a little wasteful, same as storing network inaddition to output.  
  h1.fprime.topRows(sampleSize) = h1.output.topRows(sampleSize).array()*(1.0 - h1.output.topRows(sampleSize).array());  

  // to second hidden layer
  h2.network.topRows(sampleSize) = (h1.output.topRows(sampleSize)*W2a.W).rowwise() + W2a.b; 
  h2.output.topRows(sampleSize) = 1.0/(Eigen::exp(h2.network.topRows(sampleSize).array()*-1.0) + 1.0);  
  h2.fprime.topRows(sampleSize) = h2.output.topRows(sampleSize).array()*(1.0 - h2.output.topRows(sampleSize).array());  

  // to output layer
  yPred.network.topRows(sampleSize) = (h2.output.topRows(sampleSize)*W3.W).rowwise() + W3.b; 
  yPred.output.topRows(sampleSize) = 1.0/(Eigen::exp(yPred.network.topRows(sampleSize).array()*-1.0) + 1.0);  
  yPred.fprime.topRows(sampleSize) = yPred.output.topRows(sampleSize).array()*(1.0 - yPred.output.topRows(sampleSize).array());  
}

void CS4793::PA2_FANN::klDiv(int sampleSize) {
	Eigen::RowVectorXd hmean = h1.output.colwise().mean();
	Eigen::RowVectorXd dkldiv = -srate / hmean.array() + (1-srate) / (1.0 - hmean.array());
	for (int i = 0; i < sampleSize; i++) {
		h1.error.row(i) = kl_beta * dkldiv;
	}
}


void CS4793::PA2_FANN::setErrors(int sampleSize){
   // output layer errors 
   yPred.error.topRows(sampleSize) = yTargets.topRows(sampleSize) - yPred.output.topRows(sampleSize);						//BP_STEP_1:delta_t = target - prediction (x_t) 

   // if using sparse encoding, add error component to h1.error and/or h2.error 

   // weight regularization  (or decay) : d/dw  -(1/2)beta*W_ij^2  --> -beta * W_ij
   if (useReg) {   
      // only applying to weights, not bias 
      W1a.DeltaW = W1a.W * (-beta); 
      W2a.DeltaW = W2a.W * (-beta); 
      W3.DeltaW = W3.W * (-beta); 
   }
}
// do backpropagation.  Assumes forwardPass and setErrors have all ready beed done to the network.
void CS4793::PA2_FANN::backwardPass(int sampleSize) {
   // output layer
   yPred.delta.topRows(sampleSize) = yPred.error.topRows(sampleSize).array() * yPred.fprime.topRows(sampleSize).array();	//BP_STEP_3: delta_t * f'_t(net_t)
   W3.DeltaW += lrate * h2.output.topRows(sampleSize).transpose()* yPred.delta.topRows(sampleSize)/sampleSize;				//BP_STEP_4: for all k in incoming links (in_t) and 
   W3.DeltaB += lrate * yPred.delta.topRows(sampleSize).colwise().sum()/sampleSize;

   // second hidden layer
   // if there is an error term, such as sparsity, we would add it to delta here 
   
   // add the propagation of outer layers delta's.  
   h2.delta.topRows(sampleSize) = (yPred.delta.topRows(sampleSize)*W3.W.transpose()).array()								//BP_STEP_2: for all outgoing links multiply be delta_k and add it to delta_t
                                 * h2.fprime.topRows(sampleSize).array(); 
   W2a.DeltaW += lrate * h1.output.topRows(sampleSize).transpose()* h2.delta.topRows(sampleSize) /sampleSize;
   W2a.DeltaB += lrate * h2.delta.topRows(sampleSize).colwise().sum()/sampleSize;
   
   // first hidden layer
   // if there is an error term, such as sparsity, we would add it to delta here

   // add the propagation of second hidden layers delta's.  
   h1.delta.topRows(sampleSize) = (h2.delta.topRows(sampleSize)*W2a.W.transpose()).array()
                                 * h1.fprime.topRows(sampleSize).array(); 
   W1a.DeltaW += lrate * xInput.output.topRows(sampleSize).transpose()* h1.delta.topRows(sampleSize)/sampleSize;
   W1a.DeltaB += lrate * h1.delta.topRows(sampleSize).colwise().sum()/sampleSize;
}

// do backpropagation.  Assumes forwardPass and setErrors have all ready beed done to the network.
void CS4793::PA2_FANN::preTrainBackwardPass(int sampleSize) {
	// output layer
	yPred.delta.topRows(sampleSize) = yPred.error.topRows(sampleSize).array() * yPred.fprime.topRows(sampleSize).array();	//BP_STEP_3: delta_t * f'_t(net_t)
	W3.DeltaW += lrate * h2.output.topRows(sampleSize).transpose()* yPred.delta.topRows(sampleSize) / sampleSize;				//BP_STEP_4: for all k in incoming links (in_t) and 
	W3.DeltaB += lrate * yPred.delta.topRows(sampleSize).colwise().sum() / sampleSize;

	// second hidden layer
	// if there is an error term, such as sparsity, we would add it to delta here 

	// add the propagation of outer layers delta's.  
	h2.delta.topRows(sampleSize) = (yPred.delta.topRows(sampleSize)*W3.W.transpose()).array()								//BP_STEP_2: for all outgoing links multiply be delta_k and add it to delta_t
		* h2.fprime.topRows(sampleSize).array();
	W2a.DeltaW += lrate * h1.output.topRows(sampleSize).transpose()* h2.delta.topRows(sampleSize) / sampleSize;
	W2a.DeltaB += lrate * h2.delta.topRows(sampleSize).colwise().sum() / sampleSize;

	// first hidden layer
	// if there is an error term, such as sparsity, we would add it to delta here

	// add the propagation of second hidden layers delta's.  
	h1.delta.topRows(sampleSize) = (h2.delta.topRows(sampleSize)*W2a.W.transpose()).array()
		* h1.fprime.topRows(sampleSize).array();
	W1a.DeltaW += lrate * xInput.output.topRows(sampleSize).transpose()* h1.delta.topRows(sampleSize) / sampleSize;
	W1a.DeltaB += lrate * h1.delta.topRows(sampleSize).colwise().sum() / sampleSize;
}


// do backpropagation.  Assumes forwardPass and setErrors have all ready beed done to the network.
void CS4793::PA2_FANN::altBackwardPass(int sampleSize){
   // use multiple for loops instead of matrix multiplications, hopefully helps expain the
   // calculation if having difficult with the Eigen syntax and matrices.

   double lr = lrate / sampleSize;  // include batchsize into learning rate  
   // output layer
   for (int b = 0; b < sampleSize; ++b) { // for each sample  in mini-batch ... stored in rows of most matrices (nodes)
      for(int t=0; t< yPred.delta.cols(); ++t) { // for each node in the output layer
         yPred.delta(b,t) = yPred.error(b,t) * yPred.fprime(b,t);  // place error at node t into t's delta
         W3.DeltaB(t) += lr * yPred.delta(b,t);    // bias does not need h2's output, always uses 1.0
         for (int k=0; k< h2.output.cols(); ++k) { // for each node in the 2nd hidden layer
            W3.DeltaW(k,t) += lr * h2.output(b,k) * yPred.delta(b,t);  // update incoming weights
         }
      }
   }

   // second hidden layer
   // if there is an error term, such as sparsity, we would add it to h2's delta 
   
   // propagation of outer layers delta's to 2nd layer
   for (int b = 0; b < sampleSize; ++b) { // for each sample ... rows of most matrices
      for(int t=0; t< h2.delta.cols(); ++t) { // for each node in the 2nd hidden layer
         for (int k=0; k< yPred.output.cols(); ++k) { // for each node in the outer layer
            h2.delta(b,t) += yPred.delta(b,k) * W3.W(t,k) * h2.fprime(b,t);  
         }
      }
   }

   // apply to incoming weights to second hidden layer  
   for (int b = 0; b < sampleSize; ++b) { // for each sample ... rows of most matrices
      for(int t=0; t< h2.delta.cols(); ++t) { // for each node in the 2nd hidden layer
         W2a.DeltaB(t) += lr * h2.delta(b,t);  
         for (int k=0; k< h1.output.cols(); ++k) { // for each node in the first hidden layer
            W2a.DeltaW(k,t) += lr * h1.output(b,k) * h2.delta(b,t);  
         }
      }
   }

   // first hidden layer
   // if there is an error term, such as sparsity, we would add it to delta 

   // propagation of second hidden layers delta's to first layer 
   for (int b = 0; b < sampleSize; ++b) { // for each sample ... rows of most matrices
      for(int t=0; t< h1.delta.cols(); ++t) { // for each node in the first hidden layer
         for (int k=0; k< h2.output.cols(); ++k) { // for each node in the second layer
            h1.delta(b,t) += h2.delta(b,k)*W2a.W(t,k) * h1.fprime(b,t);
         }
      }
   }

   // apply to incoming weights to first hidden layer  
   for (int b = 0; b < sampleSize; ++b) { // for each sample ... rows of most matrices
      for(int t=0; t< h1.delta.cols(); ++t) { // for each node in the first hidden layer
         W1a.DeltaB(t) += lr * h1.delta(b,t);
         for (int k=0; k< xInput.output.cols(); ++k) { // for each node in the input layer
            W1a.DeltaW(k,t) += lr * xInput.output(b,k) * h1.delta(b,t);
         }
      }
   }
}



void CS4793::PA2_FANN::updateWeights(int sampleSize){
   // learning rate and weight decay rate were applied when adding to the delta, so don't need to do it here.
   
   W3.W += W3.DeltaW;
   W3.b += W3.DeltaB;

   W2a.W += W2a.DeltaW;
   W2a.b += W2a.DeltaB;

   W1a.W += W1a.DeltaW;
   W1a.b += W1a.DeltaB;
}


/* use current ANN to predict targets for data
 *
 * Current contents of outTargets is discarded and results are returned in it.
 */
void CS4793::PA2_FANN::predict(Eigen::MatrixXd &data, Eigen::MatrixXd &outTargets) 
{  
   outTargets.resize(data.rows(),numClasses);
   // loop through each mini-batch and copy predicted output to outTargets
   int bpe = int(std::ceil(data.rows()/double(batchsize))); // number batches per epoch
   for (int j=0; j < bpe; ++j) { // for each mini-batch j
      // copy batch data to input
      int baseIndex =j*batchsize;
      int amount =std::min(batchsize, int(data.rows())-baseIndex);
      xInput.output.topRows(amount) = data.block(baseIndex,0,amount,data.cols());
      // activate network over input
      forwardPass(amount);
      outTargets.block(baseIndex,0,amount,yPred.output.cols()) = yPred.output.topRows(amount);
   }
}
// evaluate predictions on data against desired values in targets
CS4793::ErrorMeasures CS4793::PA2_FANN::evaluate(Eigen::MatrixXd &data, Eigen::MatrixXd &targets) 
{  
   // note: mean square error and classification error could be calculated after calling
   // predict on data, but also want to calc gradient information so repeating the 
   // mini-batch processing code.

   // adds error measurement components to the predict method
   int bpe = int(std::ceil(data.rows()/double(batchsize))); // number batches per epoch
   double mse = 0.0;
   int errorCount = 0;
   double meanFP1 = 0; // extra debugging info
   double meanFP2 = 0; // extra debugging info
   double meanFP3 = 0; // extra debugging info
   int yploc=0;
   int ytloc=0;
   for (int j=0; j < bpe; ++j) { // for each mini-batch j
      // copy batch data to input
      int baseIndex =j*batchsize;
      int amount =std::min(batchsize, int(data.rows())-baseIndex);
      xInput.output.topRows(amount) = data.block(baseIndex,0,amount,data.cols());
      yTargets.topRows(amount) = targets.block(baseIndex,0,amount,targets.cols());

      // activate network over input
      forwardPass(amount);

      mse  += (yTargets.topRows(amount) - yPred.output.topRows(amount)).array().square().mean() * (amount/batchsize);
      meanFP1 = h1.fprime.array().abs().mean() * (amount/batchsize);
      meanFP2 = h2.fprime.array().abs().mean() * (amount/batchsize);
      meanFP3 = yPred.fprime.array().abs().mean() * (amount/batchsize);
      //  classification count
      for (int k=0; k<amount; ++k) {
         yTargets.row(k).maxCoeff(&ytloc);  // true class
         yPred.output.row(k).maxCoeff(&yploc); // predicted class
         if (yploc != ytloc) {
            errorCount +=1;
         }
      }
   }
   mse /= bpe; 
   CS4793::ErrorMeasures res;
   res.meanSquareError = mse;
   res.classificationErrorRate = errorCount/double(data.rows()); 
   res.meanAbsLayer1FPrime= meanFP1; 
   res.meanAbsLayer2FPrime= meanFP2; 
   res.meanAbsLayer3FPrime= meanFP3; 
   return res; 
}


