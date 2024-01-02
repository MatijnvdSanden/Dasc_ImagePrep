import cv2

def addtobase(img1, img_changed):
    height, width = img1.shape[:2]
    start_col = width // 2
    img_changed = img_changed[:height, :start_col]
    img1[:, start_col:] = img_changed
    return img1

fullim = "maps\\maps\\tmp\\1.jpg"
halfim = "maps\\maps\\tmp\\UntitledOut.png"
towrite = "maps\\maps\\tmp\\1_Out.jpg"
imf = cv2.imread(fullim)
imh = cv2.imread(halfim)

out = addtobase(imf, imh)
cv2.imwrite(towrite, out)