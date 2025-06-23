import cv2
import cv2.aruco as aruco
import numpy as np
import os

def loadAugImages(path):
    """
    :param path: folder in which all the marker images are stored
    :return: dictionary with key as the id and values as the augmented image
    """
    myList = os.listdir(path)
    noOfImages = len(myList)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(os.path.join(path, imgPath))
        augDics[key] = imgAug
    return augDics

def findArucoMarker(img,markerSize=6,totalMarkers=250,draw=True):
    """
    :param img: image in which to find the aruco markers
    :param markerSize: the size of the markers
    :param totalMarkers: total number of markers that compose the dictionary
    :param draw: flag to draw bbox around markers detected
    :return: bounding boxes and id numbers of the markers detected
    """
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters()
    bboxs,ids,rejected = aruco.ArucoDetector(arucoDict,arucoParam).detectMarkers(imgGray)

    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img,bboxs)

    return [bboxs,ids]

def augmentAruco(bbox,id,img,imgAug,drawId=True):
    """
    :param bbox: the four corner points of the box
    :param id: marker id of the corresponding box used only for display
    :param img: the final image on which to draw
    :param imgAug: the image that will be overlapped on the marker
    :param drawId: flag to display the id of the detected marker
    :return: image with the augmented image overlaid
    """
    tl = (int(bbox[0][0][0]),int(bbox[0][0][1]))
    tr = (int(bbox[0][1][0]),int(bbox[0][1][1]))
    br = (int(bbox[0][2][0]),int(bbox[0][2][1]))
    bl = (int(bbox[0][3][0]),int(bbox[0][3][1]))

    h,w,c = imgAug.shape

    pts1 = np.array([tl,tr,br,bl])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    matrix,_ = cv2.findHomography(pts2,pts1)
    imgOut = cv2.warpPerspective(imgAug,matrix,(img.shape[1],img.shape[0]))
    cv2.fillConvexPoly(img,pts1.astype(int),(0,0,0))
    imgOut = img + imgOut

    if drawId:
        cv2.putText(imgOut,str(id),tl,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

    return imgOut

def main():
    cap = cv2.VideoCapture(0)
    augDics = loadAugImages("Markers")

    while True:
        success, img = cap.read()
        arucoFind = findArucoMarker(img)

        # Loop through all the markers and augment each one
        if len(arucoFind[0]) != 0:
            for bbox,id in zip(arucoFind[0],arucoFind[1]):
                if int(id) in augDics.keys():
                    img = augmentAruco(bbox,id,img,augDics[int(id)])

        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()



