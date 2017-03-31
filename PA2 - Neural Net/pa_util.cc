/* helper code for pa2.
 *
 * Feel free to use as needed for C++ implementation of PA2.
 */

// Doug Heisterkamp
// last modified: 10/17/16




#include"pa_util.h"
#include<sstream>  // for istringstream

namespace CS4793 
{
   /*! readDataAsVecOfVecs read whitespace separated reals from a stream an store into data as vector of vectors. 
    * Each line is placed as a separate row.  
    * Existing content of data is discraded.
    * returns true if all rows read same number of reals. 
    */
   bool readDataAsVecOfVecs(std::istream &in,std::vector<std::vector<double> > &data){
      data.clear(); // discard current contents
      std::string buf;
      std::istringstream line;  // for line processing
      std::vector<double> tmpVec;  // to hold row of data
      int numCols = -1;   // -1 to mean not set
      bool flag = true;
      while (getline(in,buf))  // read one line into buf
      {   
         line.clear();  // reset state of stream .. previous iteration consumed entire line
         line.str(buf); // wrap string as an istream
         double tmpD = 0.0;
         tmpVec.clear();
         while (line)  
         {     
            if (line>> tmpD)
                    tmpVec.push_back(tmpD);
         }
         if (numCols< 0){
            numCols = int(tmpVec.size());
         } else {
            if (numCols != int(tmpVec.size())) {
               flag = false;
            }
         }
         data.push_back(tmpVec); 
      }
      return (flag && numCols != -1);
   }

                                  

   /*! loadDataFile reads as much data as possible from input stream.  Assumes data is whitespace separated values
    * with each row representing one sample.  Expects the first value in each line to be the class label (which is 
    * converted to an integer).  
    *
    * params inTarget and outTarget are used for 1-hot encoding (i.e., used  1, 0 or 0.9, 0.1). 
    *  
    *
    *  Existing contents of data, oneHotTargets,tags, uniqueTags, and tag2node are discarded.
    *
    */
   bool loadDataFile(std::istream &in, 
         Eigen::MatrixXd &data,      // returns each line of stream in as row of data (with class label removed)
         Eigen::MatrixXd &oneHotTargets,  // one hot encoding of class labels, oneHotTargets.row(i) is for data.row(i)
         double inTarget,   // value to use to true class in 1-hot code
         double outTarget,   // value to use for wrong classes in 1-hot code
         std::vector<int> &tags,  // class tags of data, tags[i] is for data.row(i)
         std::set<int>  &uniqueTags,  // set of class tags read from input stream
         std::map<int,int> &tag2node  // dictionary to map tags to location in 1-hot code 
         )
   {
      tags.clear();
      uniqueTags.clear();
      tag2node.clear();

      std::vector<std::vector<double> > tmpD;
      if (! readDataAsVecOfVecs(in,tmpD)) {
         std::cerr << "Failed to read a matrix of values from stream." << std::endl;
         return false;
      }

      if (tmpD.size()==0 or tmpD[0].size() <= 1 ) {
         std::cerr << "Failed to read enough data" << std::endl;
         return false;
      }

      int numRows = int(tmpD.size());
      int numCols = int(tmpD[0].size())-1; // assuming first element is class label

      // copy into Eigen Matrix
      data.resize(numRows,numCols);
      for(int r = 0; r < numRows; ++r) {
         for(int c=0; c < numCols; ++c) {
            data(r,c) = tmpD[r][c+1];  // first element in tmpD is class tag, so use c+1 to skip
         }
      }
      
      // copy out class labels 
      tags.resize(numRows);
      for(int r=0; r < numRows; ++r) {
         tags[r] = int(tmpD[r][0]);
         uniqueTags.insert(tags[r]);
      }
      int numClasses = int(uniqueTags.size());
      for(int t : uniqueTags) {
         int loc = int(tag2node.size());
         tag2node[t] = loc;
      }
      assert(tag2node.size()==uniqueTags.size());

      //and create one hot representation
      oneHotTargets.resize(numRows,numClasses);
      oneHotTargets.fill(outTarget); // most are outTarget values
      for(int r=0; r<numRows; ++r){
         auto p = tag2node.find(tags[r]);
         assert(p != tag2node.end()); // should not happen as all tags should be placed in tag2node
         oneHotTargets(r, p->second) = inTarget;
      }
   return true;
   }

   bool loadDataFile(std::istream& in, Eigen::MatrixXd& data, Eigen::MatrixXd& oneHotTargets, int start_col, double inTarget, double outTarget) {
	   std::vector<int> tags;
	   std::set<int> uniqueTags;
	   std::map<int, int> tag2node;

	   std::vector<std::vector<double> > tmpD;
	   if (!readDataAsVecOfVecs(in, tmpD)) {
		   std::cerr << "Failed to read a matrix of values from stream." << std::endl;
		   return false;
	   }

	   std::cout << tmpD.size() << '\n';

	   if (tmpD.size() == 0 || tmpD[0].size() <= 1) {
		   std::cerr << "Failed to read enough data" << std::endl;
		   return false;
	   }

	   int numRows = int(tmpD.size());
	   int numCols = int(tmpD[0].size()) - start_col; // assuming first element is class label

											  // copy into Eigen Matrix
	   data.resize(numRows, numCols);
	   for (int r = 0; r < numRows; ++r) {
		   for (int c = 0; c < numCols; ++c) {
			   data(r, c) = tmpD[r][c + start_col];  // first element in tmpD is class tag, so use c+1 to skip
		   }
	   }

	   // copy out class labels 
	   tags.resize(numRows);
	   for (int r = 0; r < numRows; ++r) {
		   tags[r] = int(tmpD[r][0]);
		   uniqueTags.insert(tags[r]);
	   }
	   int numClasses = int(uniqueTags.size());
	   for (int t : uniqueTags) {
		   int loc = int(tag2node.size());
		   tag2node[t] = loc;
	   }
	   assert(tag2node.size() == uniqueTags.size());

	   //and create one hot representation
	   oneHotTargets.resize(numRows, numClasses);
	   oneHotTargets.fill(outTarget); // most are outTarget values
	   for (int r = 0; r<numRows; ++r) {
		   auto p = tag2node.find(tags[r]);
		   assert(p != tag2node.end()); // should not happen as all tags should be placed in tag2node
		   oneHotTargets(r, p->second) = inTarget;
	   }
	   return true;

   }
};

