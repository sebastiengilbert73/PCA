import numpy
import pickle
import PIL.Image


class PCAModel:
    def __init__(self, dataArr, zeroThreshold=0.000001):
        self.dataShape = dataArr.shape # (rows, columns)
        if self.dataShape[0] < self.dataShape[1]:
            self.smallestSizeAxis = 0
        else:
            self.smallestSizeAxis = 1


        # Compute the average vector
        self.average = dataArr.mean(axis=0)

        # Remove the average
        zeroCenteredDataArr = dataArr - self.average
        if self.smallestSizeAxis == 0:
            zeroCenteredDataArr = zeroCenteredDataArr.T
            # Instead of computing the eigenvalue decomposition of X^T X [D x D] (big),
            # We'll compute the eigenvalue decomposition of X X^T [N x N] (small).
            # X X^T v'_k = lambda_k * v'_k    : (lambda_k, v'_k) is the k'th eigenpair of X X^T
            # Pre-multiplying by X^T:
            # X^T X X^T v'_k = lambda_k * X^T v'_k
            # (X^T X) (X^T v'_k) = lambda_k (X^T v'_k)
            # -> X^T v'_k is an eigenvector of X^T X, with eigenvalue lambda_k

        self.covarianceMtx = numpy.matmul(zeroCenteredDataArr.T, zeroCenteredDataArr)

        eigenvalues, eigenvectors = numpy.linalg.eigh(self.covarianceMtx) # eigh() is specifically for a symmetric matrix (hermitian)
        self.eigenpairs = [[numpy.abs(eigenvalues[i]), eigenvectors[:,i].real] for i in range(len(eigenvalues))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        self.eigenpairs.sort(key=lambda tup: tup[0], reverse=True) # Sort with respect to the 1st item in the tuple

        # If an eigenvalue is smaller than zeroThreshold, eliminate the pair
        eigenpairsToRemove = []
        for eigenNdx in range(1, len(self.eigenpairs)):
            eigenvalue = self.eigenpairs[eigenNdx][0]
            ratio = eigenvalue / self.eigenpairs[0][0]
            if ratio < zeroThreshold:
                eigenpairsToRemove.append(self.eigenpairs[eigenNdx])
        for eigenpairToRemove in eigenpairsToRemove:
            self.eigenpairs.remove(eigenpairToRemove)

        # If the data dimension is greater than the number of samples, convert back the eigenvectors
        # to have the same dimension as the data (up to this point, the eigenvectors have a length
        # equal to the number of samples). To do so, we premultiply by X^T (remember that
        # zeroCenteredDataArr has already been transposed, so zeroCenteredDataArr = X^T
        if self.smallestSizeAxis == 0:
            # zeroCenteredDataArr.shape = (D, N)
            for eigenNdx in range(len(self.eigenpairs)):
                eigenvector = self.eigenpairs[eigenNdx][1]
                transformedEigenvector = numpy.matmul(zeroCenteredDataArr, eigenvector) # (D, N) * (N [, 1 as promoted by matmul] ) = (D, 1) -> (D, )
                # Normalize to unity
                transformedEigenvector = transformedEigenvector / numpy.linalg.norm(transformedEigenvector)
                self.eigenpairs[eigenNdx][1] = transformedEigenvector

            # Enforce orthogonality: SVD: A = U D V^T; replace D with eye() (i.e. enforce the diagonal matrix D with ones)
            eigenvectorsAsColumnsMtx = self.EigenvectorsAsColumns()
            U, S, VT = numpy.linalg.svd(eigenvectorsAsColumnsMtx, full_matrices=False)
            I = numpy.eye(U.shape[1], VT.shape[0])
            orthogonalEigenvectorsAsColumnsMtx = numpy.matmul( numpy.matmul(U, I), VT)
            for eigenNdx in range(len(self.eigenpairs)):
                self.eigenpairs[eigenNdx][1] = orthogonalEigenvectorsAsColumnsMtx[:, eigenNdx]

        # Keep a record of the variance proportions
        self.varianceProportionList = []
        sum = 0;
        for [eigenvalue, eigenvector] in self.eigenpairs:
            sum += eigenvalue
        for eigenNdx in range(len(self.eigenpairs)):
            self.varianceProportionList.append(self.eigenpairs[eigenNdx][0] / sum)


    def DataLength(self):
        return self.dataShape[1] # The number of columns is the data length

    def Average(self):
        return self.average



    def Project(self, X): #, maximumNumberOfDimensions):
        if X.shape[-1] != self.dataShape[1]:
            raise ValueError("PCAModel.Project(): The shape of X ({}) is not compatible with the model expected shape ({})".format(X.shape[-1], self.dataShape[1]))
        X_averageSubtracted = X - self.AverageAsMatrix(X.shape[0])
        eigenvectorsAsColumns = self.EigenvectorsAsColumns()
        projection = numpy.matmul(X_averageSubtracted, eigenvectorsAsColumns)
        return projection

    def Reconstruct(self, projection):
        reconstruction = self.AverageAsMatrix(projection.shape[0]) + numpy.matmul(projection, self.EigenvectorsAsRows())
        return reconstruction

    def Save(self, filepath):
        with open(filepath, 'wb') as outputFile:
            pickle.dump(self, outputFile, pickle.HIGHEST_PROTOCOL)

    def Eigenpairs(self):
        return self.eigenpairs

    def EigenvectorsAsColumns(self):
        eigenvectorsMatrix = numpy.zeros((len(self.eigenpairs[0][1]), len(self.eigenpairs) ))
        for eigenNdx in range(len(self.eigenpairs)):
            eigenvectorsMatrix[:, eigenNdx] = self.eigenpairs[eigenNdx][1]
        return eigenvectorsMatrix


    def EigenvectorsAsRows(self):
        eigenvectorsMatrix = numpy.zeros( (len(self.eigenpairs), len(self.eigenpairs[0][1])) )
        for eigenNdx in range(len(self.eigenpairs)):
            eigenvectorsMatrix[eigenNdx, :] = self.eigenpairs[eigenNdx][1]
        return eigenvectorsMatrix

    def AverageAsMatrix(self, numberOfRows):
        averageAsMatrix = numpy.repeat([self.average], numberOfRows, axis=0)
        return averageAsMatrix

    def VarianceProportion(self):
        return self.varianceProportionList

    def TruncateModel(self, numberOfEigenvectorsToKeep):
        if numberOfEigenvectorsToKeep >= len(self.eigenpairs):
            return
        eigenpairsToRemove = []
        for eigenNdx in range(numberOfEigenvectorsToKeep, len(self.eigenpairs)):
            eigenpairsToRemove.append(self.eigenpairs[eigenNdx])
        for eigenpairToRemove in eigenpairsToRemove:
            self.eigenpairs.remove(eigenpairToRemove)

def Load(filepath):
    with open(filepath, 'rb') as inputFile:
        model = pickle.load(inputFile)
    return model

def DotProduct(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("DotProduct(): The lengths of the vectors ({} and {}) are not equal".format(len(v1), len(v2)))
    sum = 0.0
    for index in range(len(v1)):
        sum += v1[index] * v2[index]
    return  sum

def NumpyVectorToImage(dataVector, numberOfRows, numberOfColumns, minValue=None, maxValue=None):
    if dataVector.size != numberOfRows * numberOfColumns:
        raise ValueError("NumpyVectorToImage(): dataVector.size ({}) != numberOfRows ({}) * numberOfColumns ({})".format(dataVector.size, numberOfRows ,numberOfColumns))
    if minValue is None:
        minValue = numpy.min(dataVector)
    if maxValue is None:
        maxValue = numpy.max(dataVector)
    if maxValue == minValue:
        maxValue = maxValue + 1.0

    image = PIL.Image.new('L', (numberOfColumns, numberOfRows) )
    image.putdata( numpy.rint( 255.0 * (dataVector - minValue)/(maxValue - minValue) ) )
    return image



def main():
    print ("PCAModel.py main()")
    X = numpy.array([ [0.14, -2.3, 1.58, 1], [-1.2, 1.62, 0.76, -1], [0.1, -0.2, 0.3, -0.4] ])
    pcaModel = PCAModel(X)
    pcaModel.TruncateModel(3)
    average = pcaModel.Average()
    eigenpairs = pcaModel.Eigenpairs()
    varianceProportionList = pcaModel.VarianceProportion()

    print ("average = {}".format(average))
    print ("eigenpairs = {}".format(eigenpairs))
    print ("varianceProportionList = {}".format(varianceProportionList))

    projection = pcaModel.Project(X)
    print ("projection = {}".format(projection))

    reconstruction = pcaModel.Reconstruct(projection)
    print ("reconstruction = {}".format(reconstruction))

if __name__ == '__main__':
    main()