import cv2
import DetectChars
import DetectPlates

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False
camera = cv2.VideoCapture(0)


def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:
        print("\nerror: KNN traning was not successful\n")
        return

    while True:
        success, imgOriginalScene = camera.read()

        if imgOriginalScene is not None:
            listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
            listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

            if len(listOfPossiblePlates) != 0:
                listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
                licPlate = listOfPossiblePlates[0]

                if len(licPlate.strChars) != 0:
                    print("\nAzonosítva= " + licPlate.strChars + "\n")
            else:
                print("\nNincs detektálható rendszám!" "\n")

        key = cv2.waitKey(1)
        if key == ord("q"):
            break



if __name__ == "__main__":
    while True:
        main()
