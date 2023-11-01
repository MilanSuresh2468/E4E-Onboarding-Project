import cv2 as cv
import time

#Security camera footage of popular street
vidCap = cv.VideoCapture('aboveground.mp4') 

#initialize background subtractors - KNN and MOG2
#KNN - finds k-nearest neighbors in training data and either assigns a label based on that or an average
BS_KNN = cv.createBackgroundSubtractorKNN()
#MOG2 - models background as gaussian distributions - Creates foreground by finding deviations
BS_MOG2 = cv.createBackgroundSubtractorMOG2()

while vidCap.isOpened():
    ret, frame=vidCap.read()
    #Original
    cv.imshow('Original', frame)
  
    #KNN
    knn_FGMask=BS_KNN.apply(frame)
    cv.imshow('KNN', knn_FGMask)

    #MOG2
    mog2_FGMask=BS_MOG2.apply(frame)
    cv.imshow('MOG2', mog2_FGMask)

    if cv.waitKey(1)& 0xFF==ord('q'):
            break

#release vid capture
cv.destroyAllWindows()
vidCap.release()
