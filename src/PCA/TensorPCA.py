import numpy as np
import PCA.PCAModel.PCAModel as PCAModel
#import PCA.ImagePCAModel as ImagePCAModel
import torch
from functools import reduce
from operator import mul

class Model:
    def __init__(self, tensors_list, zeroThreshold=0.000001):
        # Create single channel arrays with c times the height, where c in the number of channels
        if len(tensors_list) == 0:
            raise ValueError('TensorPCA.Model.__init__(): The list of tensors is empty')
        self.shape = tensors_list[0].shape

        # Check if all the tensors have the same shape
        for t in tensors_list:
            if t.shape != self.shape:
                raise ValueError(
                    f"TensorPCA.Model.__init__(): A tensor has shape {t.shape} while we expect {self.shape}")

        stacked_array = np.zeros((len(tensors_list), reduce(mul, self.shape)), dtype=float)
        for row in range(len(tensors_list)):
            stacked_array[row, :] = tensors_list[row].view(-1)  # Flatten the tensor

        self.pca_model = PCAModel(stacked_array, zeroThreshold)

    def Reshape(self, vector):
        if vector.shape != reduce(mul, self.shape):
            raise ValueError(f"TensorPCA.Model.Reshape(): vector.shape ({vector.shape}) != reduce(mul, self.shape) ({reduce(mul, self.shape)})")
        if type(vector) is torch.Tensor:
            return vector.view(self.shape)
        elif type(vector) is np.ndarray:
            return vector.reshape(self.shape)
        else:
            raise NotImplementedError(f"TensorPCA.Model.Reshape(): Not implemented type '{type(vector)}'")

    def Average(self):
        return self.Reshape(self.average)

    def Eigenpairs(self):
        eigenpairs_list = []
        for eigenvalue, eigenvector in self.eigenpairs:
            eigen_tsr = torch.tensor(self.Reshape(eigenvector))
            eigenpairs_list.append((eigenvalue, eigen_tsr))