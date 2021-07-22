import argparse
import ast
import os
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('OutputDirectory', help='The directory where the generated images will be saved')
parser.add_argument('--numberOfImages', help='The number of generated images. Default: 100', type=int, default=100)
parser.add_argument('--imageSize', help="The size of images. Default: '(320, 240)'", default='(320, 240)')
parser.add_argument('--circleCenter', help="The circle center. Default: '(160, 120)'", default='(160, 120)')
parser.add_argument('--circleDiameter', help='The circle diameter. Default: 180', type=int, default=180)
parser.add_argument('--squareCenter', help="The square center. Default: '(210, 150)'", default='(210, 150)')
parser.add_argument('--squareSize', help='The square side length. Default: 120', type=int, default=120)
args = parser.parse_args()

imageSize = ast.literal_eval(args.imageSize)
circleCenter = ast.literal_eval(args.circleCenter)
squareCenter = ast.literal_eval(args.squareCenter)



def main():
    print ("generateToyImages.py main()")

    # Create the output directory
    if not os.path.exists(args.OutputDirectory):
        os.makedirs(args.OutputDirectory)

    for imageNdx in range(args.numberOfImages):
        imageFilepath = os.path.join(args.OutputDirectory, 'image' + str(imageNdx) + '.png')
        image = np.zeros((imageSize[1], imageSize[0], 3), dtype=np.uint8)
        mask = np.zeros((imageSize[1] + 2, imageSize[0] + 2), np.uint8)
        cv2.floodFill(image, mask, (imageSize[0]//2, imageSize[1]//2),
                      newVal=np.random.randint(0, 256, (3,)).tolist())
        cv2.circle(image, circleCenter, args.circleDiameter//2, color=np.random.randint(0, 256, (3,)).tolist(),
                   thickness=cv2.FILLED)

        cv2.rectangle(image, (squareCenter[0] - args.squareSize//2, squareCenter[1] - args.squareSize//2),
                     (squareCenter[0] + args.squareSize // 2, squareCenter[1] + args.squareSize // 2),
                      color=np.random.randint(0, 256, (3,)).tolist(), thickness=cv2.FILLED)

        cv2.imwrite(imageFilepath, image)


if __name__ == '__main__':
    main()
