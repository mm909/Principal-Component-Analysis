// http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
// A tutorial on Principal Components Analysis

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

const int FeatureReduce = 0;

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
			normalizedData(j,i) -= featureMeans(i);
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

	// Remove low eigen values
	MatrixXf RFV = sortedeVectors;
	cout << endl << "Row Feature Vectors:" << endl << endl;
	cout << RFV << endl;

	MatrixXf reducedRFV = RFV;
	for (int i = reducedRFV.rows()-FeatureReduce; i < reducedRFV.rows(); i++)
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
	cout << endl << "RowOriginalData:" << endl << endl;
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

void sorteigen(MatrixXf & eValues, MatrixXf & eVectors) {
	for (int i = 0; i < eValues.rows() - 1; i++) {
		for (int j = 0; j < eValues.rows() - i - 1; j++) {
			if (eValues(j, 0) < eValues(j + 1, 0)) {
				double temp = eValues(j, 0);
				eValues(j, 0) = eValues(j + 1, 0);
				eValues(j + 1, 0) = temp;
				for (int k = 0; k < eValues.rows(); k++) {
					double temp = eVectors(k, j);
					eVectors(k, j) = eVectors(k, j+1);
					eVectors(k, j + 1) = temp;
				}
			}
		}
	}
}

double covariance(MatrixXf data, int i, int j) {
	double sum = 0;
	for (int k = 0; k < 10; k++) {
		sum += (data(k,i) * data(k, j));
	}
	sum /= 9;
	return sum;
}
