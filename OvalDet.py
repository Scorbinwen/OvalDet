import numpy as np
import os
import cv2
def WithinBound(x,y,w,h):
    if x>=0 and x<h  and y>=0 and y<w:
        return True
    else:
        return False

def DFS(x,y,arr,visit,detcolor,centroid):
    """
    Depth first search the oval region.
    :param x: type:int
    :param y: type:int
    :param arr: type:np.array
    :param visit: type:np.array
    :param detcolor: a sequence (of np.array([R,G,B]))
    :param centroid: np.array([x,y,count])
    :return:None
    """
    h,w = arr.shape[:2]

    # ToDo
    visit[x,y] = True
    centroid[0] = centroid[0] + arr[x, y, -2]
    centroid[1] = centroid[1] + arr[x, y, -1]
    centroid[2] = centroid[2] + 1
    # print("Count",centroid[0],centroid[1],centroid[2])
    # ToDo

    Delta_Xs = [0,0,1,-1]
    Delta_Ys = [1,-1,0,0]
    for (delta_x,delta_y) in zip(Delta_Xs,Delta_Ys):
        x_cur = x + delta_x
        y_cur = y + delta_y
        if WithinBound(x_cur,y_cur,w,h) and visit[x_cur,y_cur]==False\
            and np.all(arr[x_cur,y_cur,:3]==detcolor):
            DFS(x_cur,y_cur,arr,visit,detcolor,centroid)

def GetMeshGrid(h,w):

    x = np.arange(0,h).reshape((h,1))
    xs = np.tile(x,w)

    y = np.arange(0, w)
    ys = np.vstack([y for i in range(h)])
    return xs,ys

def KeyPointDet(arr,DetColors,savetxtpath):
    """
    Detect keypoints
    :param arr: type:np.array()
    :param DetColors: a list of np.array([R,G,B])
    :return:
    """
    assert arr.shape[-1]==3,["Expect channels of arr should be 3,got {}".format(arr.shape[-1])]
    visit = np.zeros_like(arr[...,0],dtype=np.bool)

    h,w = arr.shape[:2]
    # Concat additional 2 channels(x-chl and y-chl) to arr
    meshx,meshy = GetMeshGrid(h,w)

    arr = np.concatenate([arr,np.expand_dims(meshx,axis=-1),np.expand_dims(meshy,axis=-1)],axis=-1)

    Centroids = []
    for detcolor in DetColors:
        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                centroid = np.array([0,0,0])
                if visit[x,y]==False and  np.all(arr[x,y,:3]==detcolor):

                    DFS(x,y,arr,visit,detcolor,centroid)
                    centroid[0] = centroid[0]/centroid[2]
                    centroid[1] = centroid[1]/centroid[2]
                    Centroids.append(centroid[:2])

    # Open file

    fd = os.open(savetxtpath, os.O_RDWR | os.O_CREAT)

    # Writing text
    print("Save keypoints to {}...".format(savetxtpath))
    for centroid in Centroids:
        print(centroid)
        ret = os.write(fd, str.encode(str(centroid)))

    os.close(fd)




def run():
    # arr = np.zeros((256,256,3))
    # DetColors = [np.array([255,0,0]),np.array([255,255,0])]
    # arr[100:110,100:110,:]=255,0,0
    #
    # arr[120:140,100:110,:]=255,255,0
    # KeyPointDet(arr,DetColors)
    DetColors = [np.array([255,0,0])]
    path2imgfolder = ""
    savetxtpath = "keypoints.txt"
    for imgfile in os.listdir(path2imgfolder):
        img_arr = cv2.imread(os.path.join(path2imgfolder,imgfile))
        KeyPointDet(img_arr,DetColors,savetxtpath)


arr = np.zeros((256,256,3))
cv2.imwrite("test.jpg",arr)
img = cv2.imread("test.jpg")
cv2.resize(img,(256,256))
