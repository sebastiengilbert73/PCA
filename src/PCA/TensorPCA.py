import numpy as np
from PCA.PCAModel import PCAModel
#import PCA.ImagePCAModel as ImagePCAModel
import torch
from functools import reduce
from operator import mul
import logging

class Model:
    def __init__(self, tensors_list, zeroThreshold=0.000001):
        # Create single channel arrays with length equal to the product of the dimensions
        if len(tensors_list) == 0:
            raise ValueError('TensorPCA.Model.__init__(): The list of tensors is empty')
        self.shape = tensors_list[0].shape

        # Check if all the tensors have the same shape
        for t in tensors_list:
            if t.shape != self.shape:
                raise ValueError(
                    f"TensorPCA.Model.__init__(): A tensor has shape {t.shape} while we expect {self.shape}")

        stacked_array = np.zeros((len(tensors_list), reduce(mul, self.shape)), dtype=float)
        #logging.debug(f"TensorPCA.Model(): stacked_array.shape = {stacked_array.shape}")
        
        for row in range(len(tensors_list)):
            stacked_array[row, :] = tensors_list[row].reshape(-1)  # Flatten the tensor

        self.pca_model = PCAModel(stacked_array, zeroThreshold)

    def Reshape(self, vector):
        if vector.shape != (reduce(mul, self.shape), ):
            raise ValueError(f"TensorPCA.Model.Reshape(): vector.shape ({vector.shape}) != reduce(mul, self.shape) ({reduce(mul, self.shape)})")
        if type(vector) is torch.Tensor:
            return vector.view(self.shape)
        elif type(vector) is np.ndarray:
            return vector.reshape(self.shape)
        else:
            raise NotImplementedError(f"TensorPCA.Model.Reshape(): Not implemented type '{type(vector)}'")

    def Average(self):
        return self.Reshape(self.pca_model.average)

    def Eigenpairs(self):
        eigenpairs_list = []
        for eigenvalue, eigenvector in self.pca_model.eigenpairs:
            eigen_tsr = torch.tensor(self.Reshape(eigenvector))
            eigenpairs_list.append((eigenvalue, eigen_tsr))
        return eigenpairs_list

    def Project(self, input_tsr):  # input_tsr.shape = (C, H, W)
        if input_tsr.shape != self.shape:
            raise ValueError(f"TensorPCA.Model.Project(): input_tsr.shape ({input_tsr.shape}) != self.shape ({self.shape})")
        input_vct = input_tsr.reshape(-1)
        #print(f"TensorPCA.Project(): input_vct.shape = {input_vct.shape}")
        projection = self.pca_model.Project(input_vct.unsqueeze(0))
        return projection