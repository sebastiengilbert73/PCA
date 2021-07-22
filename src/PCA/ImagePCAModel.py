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


class MultiChannelModel:
    def __init__(self, imagesList, zeroThreshold=0.000001):
        # Create monochrome images with c times the height, where c in the number of channels
        if len(imagesList) == 0:
            raise ValueError('ImagePCAModel.MultiChannelModel.__init__(): The list of images is empty')
        self.image_shapeHWC = imagesList[0].shape
        self.image_sizeHW = (self.image_shapeHWC[0], self.image_shapeHWC[1])
        self.number_of_channels = self.image_shapeHWC[2]

        # Check if all the images have the same size
        for image in imagesList:
            if image.shape != self.image_shapeHWC:
                raise ValueError("ImagePCAModel.MultiChannelModel.__init__(): An image has shape {} while we expect {}".format(image.shape, self.image_shapeHWC))

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
        height = self.image_sizeHW[0]
        unstacked_img = np.zeros(self.image_shapeHWC, dtype=np.uint8)
        for channelNdx in range(self.number_of_channels):
            starting_y = channelNdx * height
            unstacked_img[:, :, channelNdx] = stacked_image[starting_y: starting_y + height, :]
        return unstacked_img

    def Stack(self, color_image):
        height = self.image_sizeHW[0]

        stacked_image = np.zeros((self.number_of_channels * height, self.image_sizeHW[1]), dtype=np.uint8)
        for channelNdx in range(self.number_of_channels):
            starting_y = channelNdx * height
            stacked_image[starting_y: starting_y + height, :] = color_image[:, :, channelNdx]
        return stacked_image

    def TruncateModel(self, numberOfEigenvectorsToKeep):
        self.stacked_pca_model.TruncateModel(numberOfEigenvectorsToKeep)

    def Project(self, image):
        if image.shape != self.image_shapeHWC:
            raise ValueError("ImagePCAModel.MultiChannelModel.Project(): The image shape {} doesn't match the model expected shape {}".format(image.shape, self.image_shapeHWC))
        stacked_img = self.Stack(image)
        return self.stacked_pca_model.Project(stacked_img)

    def Reconstruct(self, projection):
        dataArr = self.stacked_pca_model.Reconstruct(projection)
        stacked_reconstruction = np.reshape(dataArr, (self.number_of_channels * self.image_sizeHW[0], self.image_sizeHW[1]))
        return self.Unstack(stacked_reconstruction)

    def VarianceProportion(self):
        return self.stacked_pca_model.VarianceProportion()

    def Eigenpairs(self):
        return self.stacked_pca_model.Eigenpairs()


def Load(filepath):
    with open(filepath, 'rb') as inputFile:
        model = pickle.load(inputFile)
    return model

