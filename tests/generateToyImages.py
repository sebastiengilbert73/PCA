import argparse
import ast
#import PIL.Image
#import PIL.ImageDraw
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

    for imageNdx in range(args.numberOfImages):
        imageFilepath = os.path.join(args.OutputDirectory, 'image' + str(imageNdx) + '.png')
        image = np.ones((imageSize[1], imageSize[0]), dtype=np.uint8) * np.random.randint(256)
        cv2.circle(image, circleCenter, args.circleDiameter//2, np.random.randint(256), thickness=cv2.FILLED)

        cv2.rectangle(image, (squareCenter[0] - args.squareSize//2, squareCenter[1] - args.squareSize//2),
                     (squareCenter[0] + args.squareSize // 2, squareCenter[1] + args.squareSize // 2),
                     np.random.randint(256), thickness=cv2.FILLED)
        """rectangleGrayLevel = numpy.random.randint(256)
        for y in range(int(squareCenter[1] - args.squareSize/2), int(squareCenter[1] + args.squareSize/2)):
            for x in range(int(squareCenter[0] - args.squareSize / 2), int(squareCenter[0] + args.squareSize / 2)):
                currentGrayLevel = image.getpixel((x, y))
                newGrayLevel = (currentGrayLevel + rectangleGrayLevel) %256
                image.putpixel((x, y), value=newGrayLevel)
        image.save(imageFilepath)
        """
        cv2.imwrite(imageFilepath, image)


if __name__ == '__main__':
    main()