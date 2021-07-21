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
        # Create monochrome images with three times the height
        if len(imagesList) == 0:
            raise ValueError('ImagePCAModel.ColorModel.__init__(): The list of images is empty')
        self.imageShapeHW = imagesList[0].shape

        # Check if all the images have the same size
        for image in imagesList:
            imageHeight, imageWidth, channels = image.shape
            if (imageWidth != self.imageShapeHW[1]) or (imageHeight != self.imageShapeHW[0]):
                raise ValueError("ImagePCAModel.ColorModel.__init__(): The images do not all have the same size: ({}, {}) != ({}, {})".format(imageWidth, imageHeight, self.imageShapeHW[1], self.imageShapeHW[0]))
            if channels != 3:
                raise ValueError("ImagePCAModel.ColorModel.__init__(): An image is not color")

        stacked_images_list = []
        for image in imagesList:
            stacked_image = self.Stack(image)
            stacked_images_list.append(stacked_image)
        self.stacked_pca_model = GrayscaleModel(stacked_images_list, zeroThreshold)


    def AverageImage(self):
        stacked_average_image = self.stacked_pca_model.AverageImage()
        return self.Unstack(stacked_average_image)

    def EigenImagesForDisplay(self):
        stacked_eigenimages = self.stacked_pca_model.EigenImagesForDisplay()
        eigenimages = [self.Unstack(stacked_eigenimage) for stacked_eigenimage in stacked_eigenimages]
        return eigenimages

    def Save(self, filepath):
        with open(filepath, 'wb') as outputFile:
            pickle.dump(self, outputFile, pickle.HIGHEST_PROTOCOL)

    def Unstack(self, stacked_image):
        height = self.imageShapeHW[0]
        unstacked_img = np.zeros(self.imageShapeHW, dtype=np.uint8)
        unstacked_img[:, :, 0] = stacked_image[0: height, :]
        unstacked_img[:, :, 1] = stacked_image[height: 2 * height, :]
        unstacked_img[:, :, 2] = stacked_image[2 * height:, :]
        return unstacked_img

    def Stack(self, color_image):
        height = self.imageShapeHW[0]
        stacked_image = np.zeros((3 * height, self.imageShapeHW[1]), dtype=np.uint8)
        stacked_image[0: height, :] = color_image[:, :, 0]
        stacked_image[height: 2 * height, :] = color_image[:, :, 1]
        stacked_image[2 * height:, :] = color_image[:, :, 2]
        return stacked_image

    def TruncateModel(self, numberOfEigenvectorsToKeep):
        self.stacked_pca_model.TruncateModel(numberOfEigenvectorsToKeep)

    def Project(self, image):
        height, width, channels = image.shape
        if width != self.imageShapeHW[1] or height != self.imageShapeHW[0]:
            raise ValueError("ImagePCAModel.ColorModel.Project(): The input image size ({}, {}) doesn't match the model size ({}, {})".format(width, height, self.imageShapeHW[1], self.imageShapeHW[0]))
        if channels != 3:
            raise ValueError("ImagePCAModel.ColorModel.Project(): The input image is not color")
        stacked_img = self.Stack(image)
        return self.stacked_pca_model.Project(stacked_img)

    def Reconstruct(self, projection):
        dataArr = self.stacked_pca_model.Reconstruct(projection)
        stacked_reconstruction = np.reshape(dataArr, (3 * self.imageShapeHW[0], self.imageShapeHW[1]))
        return self.Unstack(stacked_reconstruction)

    def VarianceProportion(self):
        return self.stacked_pca_model.VarianceProportion()


def Load(filepath):
    with open(filepath, 'rb') as inputFile:
        model = pickle.load(inputFile)
    return model

