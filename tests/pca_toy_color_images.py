import argparse
import logging
import os
import cv2
import PCA.ImagePCAModel

parser = argparse.ArgumentParser()
parser.add_argument('ImagesDirectory', help='The directory where the images are')
parser.add_argument('OutputDirectory', help="The output directory")
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)


def main():
    logging.info("pca_toy_color_images.py main()")
    # Create the output directory
    if not os.path.exists(args.OutputDirectory):
        os.makedirs(args.OutputDirectory)

    filepaths = [os.path.join(args.ImagesDirectory, f) for f in os.listdir(args.ImagesDirectory) if
                 os.path.isfile(os.path.join(args.ImagesDirectory, f))
                 and f.upper().startswith('IMAGE')]

    imagesList = [cv2.imread(filepath, cv2.IMREAD_COLOR) for filepath in filepaths]

    print ("len(imagesList) = {}".format(len(imagesList)))
    imagePCAModel = PCA.ImagePCAModel.MultiChannelModel(imagesList, zeroThreshold=1e-6)
    averageImg = imagePCAModel.AverageImage()
    cv2.imwrite(os.path.join(args.OutputDirectory, 'average.png'), averageImg)

    imagePCAModel.Save(os.path.join(args.OutputDirectory, 'model.pca'))

    eigenpairs = imagePCAModel.Eigenpairs()

    eigenImagesForDisplay = imagePCAModel.EigenImagesForDisplay()
    for eigenNdx in range(len(eigenImagesForDisplay)):
        eigenvectorImgFilepath = os.path.join(args.OutputDirectory, 'eigenvector' + str(eigenNdx) + '.png')
        cv2.imwrite(eigenvectorImgFilepath, eigenImagesForDisplay[eigenNdx])

    testImage = imagesList[0]
    imagePCAModel.TruncateModel(6)
    projection = imagePCAModel.Project(testImage)
    print("projection = {}".format(projection))
    reconstruction = imagePCAModel.Reconstruct(projection)
    cv2.imwrite(os.path.join(args.OutputDirectory, 'reconstruction.png'), reconstruction)
    varianceProportion = imagePCAModel.VarianceProportion()
    print("varianceProportion = {}".format(varianceProportion))



if __name__ == '__main__':
    main()
