import argparse
import logging
import numpy as np
from PCA.PCAModel import PCAModel

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)


def main():
    logging.info("pca_toy_vector.py main()")
    X = np.array([[0.14, -2.3, 1.58, 1], [-1.2, 1.62, 0.76, -1], [0.1, -0.2, 0.3, -0.4]])
    pcaModel = PCAModel(X)
    pcaModel.TruncateModel(3)
    average = pcaModel.Average()
    eigenpairs = pcaModel.Eigenpairs()
    varianceProportionList = pcaModel.VarianceProportion()

    logging.info("average = {}".format(average))
    logging.info("eigenpairs = {}".format(eigenpairs))
    logging.info("varianceProportionList = {}".format(varianceProportionList))

    projection = pcaModel.Project(X)
    logging.info("projection = {}".format(projection))

    reconstruction = pcaModel.Reconstruct(projection)
    logging.info("reconstruction = {}".format(reconstruction))


if __name__ == '__main__':
    main()
