#ifndef __PA_UTIL_H__
#define __PA_UTIL_H__
/* helper code for pa2.
 *
 * Feel free to use as needed for C++ implementation of PA2.
 */

// Doug Heisterkamp
// last modified: 10/17/16


#include<iostream>
#include<Eigen/Core>
#include<vector>
#include<map>
#include<set>

namespace CS4793 
{
   /*! readDataAsVecOfVecs read whitespace separated reals from a stream an store into data as vector of vectors. 
    * Each line is placed as a separate row.  
    * Existing content of data is discraded.
    * returns true if all rows read same number of reals. 
    *
    * a generic helper function for loadDataFile.
    */
   bool readDataAsVecOfVecs(std::istream &in, std::vector<std::vector<double> > &data);

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
         );

   bool loadDataFile(std::istream &in,
	   Eigen::MatrixXd &data,      // returns each line of stream in as row of data (with class label removed)
	   Eigen::MatrixXd &oneHotTargets,  // one hot encoding of class labels, oneHotTargets.row(i) is for data.row(i)
	   int start_col = 0,
	   double inTarget = 0.9,   // value to use to true class in 1-hot code
	   double outTarget = 0.1   // value to use for wrong classes in 1-hot code
   );

      // todo: functions saveModelPA2 and loadModelPA2
}
#endif
