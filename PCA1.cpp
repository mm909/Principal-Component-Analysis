// http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
// A tutorial on Principal Components Analysis

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

const int FeatureReduce = 1;

double covariance(MatrixXf, int, int);
void sorteigen(MatrixXf&, MatrixXf&);
void removeRow(MatrixXf&, unsigned int);

int main() {

	// Assign data
	// { 2.5, 2.4 }
	// { 0.5, 0.7 }
	// { 2.2, 2.9 }
	// { 1.9, 2.2 }
	// { 3.1, 3.0 }
	// { 2.3, 2.7 }
	// { 2.0, 1.6 }
	// { 1.0, 1.1 }
	// { 1.5, 1.6 }
	// { 1.1, 0.9 }
	// { 1.1, 0.9 }

	MatrixXf data(10, 2);
	data(0, 0) = 2.5;
	data(0, 1) = 2.4;
	data(1, 0) = 0.5;
	data(1, 1) = 0.7;
	data(2, 0) = 2.2;
	data(2, 1) = 2.9;
	data(3, 0) = 1.9;
	data(3, 1) = 2.2;
	data(4, 0) = 3.1;
	data(4, 1) = 3.0;
	data(5, 0) = 2.3;
	data(5, 1) = 2.7;
	data(6, 0) = 2.0;
	data(6, 1) = 1.6;
	data(7, 0) = 1.0;
	data(7, 1) = 1.1;
	data(8, 0) = 1.5;
	data(8, 1) = 1.6;
	data(9, 0) = 1.1;
	data(9, 1) = 0.9;
	data(9, 0) = 1.1;
	data(9, 1) = 0.9;

	cout << endl << "Starting Data: " << endl << endl;
	cout << data << endl;

	cout << endl << "============================================" << endl;

	// Calc Feature Means
	MatrixXf featureMeans(2, 1);
	MatrixXf featureIdenity(1, 10);
	for (int i = 0; i < 2; i++) {
		featureMeans(i, 0) = 0;
	}

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 10; j++) {
			featureMeans(i, 0) += data(j, i);
			featureIdenity(0, j) = 1;
		}
		featureMeans(i, 0) = featureMeans(i, 0) / 10;
	}

	// Normalize data
	MatrixXf normalizedData(10, 2);
	normalizedData = data;
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 10; j++) {
			normalizedData(j, i) -= featureMeans(i);
		}
	}

	cout << endl << "Normalized Data: " << endl << endl;
	cout << normalizedData << endl;

	// Transpose Normalized data
	MatrixXf transposedNormalizedData(10, 2);
	transposedNormalizedData = normalizedData;
	cout << endl << "Transposed Normalized Data:" << endl << endl;
	transposedNormalizedData.transposeInPlace();
	cout << transposedNormalizedData << endl;

	cout << endl << "============================================" << endl;

	// Compute the Covariance Matrix
	MatrixXf covarMatrix(2, 2);
	covarMatrix(0, 0) = covariance(normalizedData, 0, 0);
	covarMatrix(0, 1) = covariance(normalizedData, 0, 1);
	covarMatrix(1, 0) = covariance(normalizedData, 1, 0);
	covarMatrix(1, 1) = covariance(normalizedData, 1, 1);

	cout << endl << "Covariance Matrix: " << endl << endl;
	cout << covarMatrix << endl;

	cout << endl << "============================================" << endl;

	// Compute eigen values and vectors
	SelfAdjointEigenSolver<Matrix2f> eigensolver(covarMatrix);
	if (eigensolver.info() != Success) abort();

	MatrixXf eValues = eigensolver.eigenvalues();
	cout << endl << "Eigenvalues: " << endl << endl;
	cout << eValues << endl;

	MatrixXf eVectors = eigensolver.eigenvectors();
	cout << endl << "Eigenvectors: " << endl << endl;
	cout << eVectors << endl;

	cout << endl << "============================================" << endl;

	// Sort eigen values
	MatrixXf sortedEValues = eigensolver.eigenvalues();
	MatrixXf sortedeVectors = eigensolver.eigenvectors();
	sorteigen(sortedEValues, sortedeVectors);

	cout << endl << "Sorted Eigen Values:" << endl << endl;
	cout << sortedEValues << endl;

	cout << endl << "Sorted Eigen Vectors:" << endl << endl;
	cout << sortedeVectors << endl;

	cout << endl << "============================================" << endl;

	double sumEValues = 0;
	MatrixXf representation = sortedEValues;
	for (int i = 0; i < representation.rows(); i++) {
		sumEValues += representation(i, 0);
	}
	for (int i = 0; i < representation.rows(); i++) {
		representation(i, 0) /= sumEValues;
	}

	cout << endl << "Representation of the data for each feature:" << endl << endl;
	cout << representation << endl;
	

	cout << endl << "============================================" << endl;

	// Remove low eigen values
	MatrixXf RFV = sortedeVectors;
	cout << endl << "Row Feature Vectors:" << endl << endl;
	cout << RFV << endl;

	MatrixXf reducedRFV = RFV;
	for (int i = reducedRFV.rows() - FeatureReduce; i < reducedRFV.rows(); i++)
		for (int j = 0; j < reducedRFV.cols(); j++)
			reducedRFV(i, j) = 0;
	cout << endl << "Reduced Row Feature Vector: " << endl << endl;
	cout << reducedRFV << endl;

	cout << endl << "============================================" << endl;

	// Get final data values
	cout << endl << "Final Data:" << endl << endl;
	MatrixXf Final = reducedRFV * transposedNormalizedData;
	cout << Final.transpose() << endl;

	// Get origional databack
	// RowOriginalData = (RowFeatureVector^T x FinalData) + OriginalMean;
	reducedRFV.transposeInPlace();
	cout << endl << "Reconstructed Data:" << endl << endl;
	MatrixXf newData = (reducedRFV * Final) + (featureMeans * featureIdenity);
	cout << newData.transpose() << endl;

	cout << endl << "============================================" << endl;
}

void removeRow(MatrixXf& matrix, unsigned int rowToRemove) {
	unsigned int numRows = matrix.rows() - 1;
	unsigned int numCols = matrix.cols();

	if (rowToRemove < numRows) {
		matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);
	}

	matrix.conservativeResize(numRows, numCols);
}

void sorteigen(MatrixXf& eValues, MatrixXf& eVectors) {
	for (int i = 0; i < eValues.rows() - 1; i++) {
		for (int j = 0; j < eValues.rows() - i - 1; j++) {
			if (eValues(j, 0) < eValues(j + 1, 0)) {
				double temp = eValues(j, 0);
				eValues(j, 0) = eValues(j + 1, 0);
				eValues(j + 1, 0) = temp;
				for (int k = 0; k < eValues.rows(); k++) {
					double temp = eVectors(k, j);
					eVectors(k, j) = eVectors(k, j + 1);
					eVectors(k, j + 1) = temp;
				}
			}
		}
	}
}

double covariance(MatrixXf data, int i, int j) {
	double sum = 0;
	for (int k = 0; k < 10; k++) {
		sum += (data(k, i) * data(k, j));
	}
	sum /= 9;
	return sum;
}
/*


Starting Data:

2.5 2.4
0.5 0.7
2.2 2.9
1.9 2.2
3.1   3
2.3 2.7
  2 1.6
  1 1.1
1.5 1.6
1.1 0.9

============================================

Normalized Data:

	 0.69      0.49
	-1.31     -1.21
	 0.39      0.99
0.0899999      0.29
	 1.29      1.09
	 0.49      0.79
	 0.19     -0.31
	-0.81     -0.81
	-0.31     -0.31
	-0.71     -1.01

Transposed Normalized Data:

	 0.69     -1.31      0.39 0.0899999      1.29      0.49      0.19     -0.81     -0.31     -0.71
	 0.49     -1.21      0.99      0.29      1.09      0.79     -0.31     -0.81     -0.31     -1.01

============================================

Covariance Matrix:

0.616556 0.615444
0.615444 0.716556

============================================

Eigenvalues:

0.0490834
  1.28403

Eigenvectors:

-0.735179 -0.677873
 0.677873 -0.735179

============================================

Sorted Eigen Values:

  1.28403
0.0490834

Sorted Eigen Vectors:

-0.677873 -0.735179
-0.735179  0.677873

============================================

Representation of the data for each feature:

 0.963181
0.0368187

============================================

Row Feature Vectors:

-0.677873 -0.735179
-0.735179  0.677873

Reduced Row Feature Vector:

-0.677873 -0.735179
		0         0

============================================

Final Data:

 -0.82797         0
  1.77758        -0
-0.992198         0
 -0.27421         0
  -1.6758         0
-0.912949         0
0.0991095         0
  1.14457        -0
 0.438046        -0
  1.22382        -0

RowOriginalData:

 2.37126  2.51871
0.605025 0.603161
 2.48258  2.63944
 1.99588  2.11159
 2.94598  3.14201
 2.42886  2.58118
 1.74282  1.83714
 1.03412  1.06853
 1.51306  1.58796
0.980405  1.01027

============================================

*/
