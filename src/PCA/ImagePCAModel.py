import numpy
import PCA.PCAModel as PCAModel
import os
#import PIL.Image
#import PIL.ImageMath

import pickle
import numpy as np
import cv2


class GrayscaleModel:
    def __init__(self, imagesList, zeroThreshold=0.000001):
        if len(imagesList) == 0:
            raise ValueError('GrayscaleModel.__init__(): The list of images is empty')
        self.imageShapeHW = imagesList[0].shape

        # Check if all the images have the same size
        for image in imagesList:
            imageHeight, imageWidth = image.shape
            if (imageWidth != self.imageShapeHW[1]) or (imageHeight != self.imageShapeHW[0]):
                raise ValueError("GrayscaleModel.__init__(): The images do not all have the same size: ({}, {}) != ({}, {})".format(imageWidth, imageHeight, self.imageShapeHW[1], self.imageShapeHW[0]))
            if len(image.shape) != 2:
                raise ValueError("GrayscaleModel.__init__(): An image is not grayscale")

        # Convert the images into an array
        dataArr = numpy.zeros( (len(imagesList), self.imageShapeHW[0] * self.imageShapeHW[1]), dtype=numpy.float64)
        for imageNdx in range(len(imagesList)):
            image = imagesList[imageNdx]
            imgData = image.flatten().astype(numpy.float64)
            dataArr[imageNdx, :] = imgData

        # Build the PCA model
        self.PCAModel = PCAModel.PCAModel(dataArr, zeroThreshold=zeroThreshold)

    def Eigenpairs(self):
        eigenpairs = []
        flattenedEigenpairs = self.PCAModel.Eigenpairs()
        for eigenNdx in range(len(flattenedEigenpairs)):
            eigenvalue = flattenedEigenpairs[eigenNdx][0]
            eigenImage = np.reshape(np.array(flattenedEigenpairs[eigenNdx][1]), self.imageShapeHW)
            eigenpairs.append([eigenvalue, eigenImage])
        return eigenpairs

    def ImageSize(self):
        return self.imageShapeHW

    def AverageImage(self):
        averageDataList = self.PCAModel.Average().astype(numpy.uint8)
        averageImg = np.reshape(averageDataList,self.imageShapeHW)
        return averageImg

    def EigenImagesForDisplay(self):
        eigenpairs = self.Eigenpairs()
        eigenImagesForDisplay = []
        for eigenNdx in range(len(eigenpairs)):
            eigenImage = eigenpairs[eigenNdx][1]
            minValue = eigenImage.min()
            maxValue = eigenImage.max()
            if maxValue - minValue <= 0:
                maxValue = minValue + 1.0
            imageForDisplay = np.reshape(np.array(255 * (eigenImage - minValue)/(maxValue - minValue)).astype(np.uint8),
                                         self.imageShapeHW)
            eigenImagesForDisplay.append(imageForDisplay)
        return eigenImagesForDisplay

    def Save(self, filepath):
        with open(filepath, 'wb') as outputFile:
            pickle.dump(self, outputFile, pickle.HIGHEST_PROTOCOL)

    def Project(self, image):
        height, width = image.shape
        if width != self.imageShapeHW[1] or height != self.imageShapeHW[0]:
            raise ValueError("ImagePCAModel.Project(): The input image size ({}, {}) doesn't match the model size ({}, {})".format(width, height, self.imageShapeHW[1], self.imageShapeHW[0]))
        if len (image.shape) != 2:
            raise ValueError("ImagePCAModel.Project(): The input image is not single-channel")
        imageDataArr = numpy.asarray(image.flatten(), dtype=numpy.float64)

        # Convert it to a 1 row matrix
        imageDataArr = imageDataArr[numpy.newaxis, :]
        return self.PCAModel.Project(imageDataArr)

    def Reconstruct(self, projection):
        dataArr = self.PCAModel.Reconstruct(projection)
        reconstruction = np.reshape(dataArr, self.imageShapeHW)
        return reconstruction

    def VarianceProportion(self):
        return self.PCAModel.VarianceProportion()

    def TruncateModel(self, numberOfEigenvectorsToKeep):
        self.PCAModel.TruncateModel(numberOfEigenvectorsToKeep)


class ColorModel:
    def __init__(self, imagesList, zeroThreshold=0.000001):
        pass

    def AverageImage(self):
        pass

    def EigenImagesForDisplay(self):
        pass

def Load(filepath):
    with open(filepath, 'rb') as inputFile:
        model = pickle.load(inputFile)
    return model

