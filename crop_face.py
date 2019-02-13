##################################################################
# This script crop faces from a folder contains many human figures
##################################################################

import sys
import dlib
import cv2
import os

Images_Folder = 'train/me'
OutFace_Folder = 'train/me_face/'

Images_Path = os.path.join(os.path.realpath('.'), Images_Folder)

pictures = os.listdir(Images_Path)

detector = dlib.get_frontal_face_detector()

print(pictures)

def rotate(img):
    rows,cols,_ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

for f in pictures:
    img = cv2.imread(os.path.join(Images_Path,f), cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    img = rotate(img)

    dets = detector(img, 1)
    #print("Number of faces detected: {}".format(len(dets)))

    for idx, face in enumerate(dets):
        # print('face{}; left{}; top {}; right {}; bot {}'.format(idx, face.left(). face.top(), face.right(), face.bottom()))

        left = face.left()
        top = face.top()
        right = face.right()
        bot = face.bottom()
        #print(left, top, right, bot)
        #cv2.rectangle(img, (left, top), (right, bot), (0, 255, 0), 3)
        #print(img.shape)
        crop_img = img[top:bot, left:right]
        #cv2.imshow(f, img)
        #cv2.imshow(f, crop_img)
        cv2.imwrite(OutFace_Folder+f[:-4]+"_face.jpg", crop_img)
        #k = cv2.waitKey(1000)
        #cv2.destroyAllWindows()


