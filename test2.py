import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys


def getroad(input):
    # Define the lower and upper bounds for the road color
    lower_road = np.array([240, 240, 240], dtype="uint8")
    upper_road = np.array([255, 255, 255], dtype="uint8")
    # Create a mask to extract the road color
    road_mask = cv2.inRange(img, lower_road, upper_road)

    # Perform morphological operations to fill enclosed regions
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)



    # Remove small white dots in the black region
    kernel_open = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    #kernel_close = np.ones((4,4), np.uint8)
    #result_road = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_dilate = np.ones((2, 2), np.uint8)  # You can adjust the kernel size as needed
    expanded_road = cv2.dilate(closed, kernel_dilate, iterations=2)

    backtorgb_road = cv2.cvtColor(expanded_road, cv2.COLOR_GRAY2RGB)
    return backtorgb_road


def gethighway(input):
    #Highway
    lower_highway = np.array([10, 70, 100], dtype="uint8")
    upper_highway = np.array([165, 255, 255], dtype="uint8")


    highway_mask = cv2.inRange(img, lower_highway, upper_highway)

    kernel_highway= np.ones((2, 2), np.uint8)
    closed_mask_highway = cv2.morphologyEx(highway_mask, cv2.MORPH_CLOSE, kernel_highway, iterations=2)
    
    # Remove small white dots in the black region
    kernel_open = np.ones((1, 1), np.uint8)
    closed = cv2.morphologyEx(closed_mask_highway, cv2.MORPH_OPEN, kernel_open, iterations=2)

    kernel_close = np.ones((1,1), np.uint8)
    result_highway = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    kernel_dilate = np.ones((2, 2), np.uint8)  # You can adjust the kernel size as needed
    expanded_highway = cv2.dilate(result_highway, kernel_dilate, iterations=5)

    backtorgb_highway = cv2.cvtColor(expanded_highway,cv2.COLOR_GRAY2RGB)

    backtorgb_highway[np.all(backtorgb_highway == (255, 255, 255), axis=-1)] = (0,255,255) #this is in bgr!
    return backtorgb_highway

def getgreen(input):

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    green_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(green_hsv, lower_green, upper_green)

    kernel_green= np.ones((1, 1), np.uint8)
    closed_mask_green = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_green)
    backtorgb_green = cv2.cvtColor(closed_mask_green,cv2.COLOR_GRAY2RGB)

    backtorgb_green[np.all(backtorgb_green == (255, 255, 255), axis=-1)] = (0,255,0) #this is in bgr!
    return backtorgb_green

def combine_highway_and_road(img_highway, img_road):
    highway_hsv = cv2.cvtColor(img_highway, cv2.COLOR_BGR2HSV)

    # Define the range of orange color in HSV
    lower_orange = np.array([10, 1, 1])
    upper_orange = np.array([50, 255, 255])

    # Create a mask for pixels in the orange range
    orange_mask = cv2.inRange(highway_hsv, lower_orange, upper_orange)

    # Apply the mask to the "backtorgb_highway" image
    orange_pixels = cv2.bitwise_and(img_highway, img_highway, mask=orange_mask)

    # Resize the mask to match the dimensions of the "backtorgb_road" image
    resized_mask = cv2.resize(orange_mask, (img_road.shape[1], img_road.shape[0]))

    # Create an inverted mask for the "backtorgb_road" image
    inv_mask = cv2.bitwise_not(resized_mask)

    # Extract the orange pixels from the "backtorgb_road" image
    road_orange_pixels = cv2.bitwise_and(img_road, img_road, mask=inv_mask)

    # Combine the extracted orange pixels and the original "backtorgb_road" image
    result = cv2.add(road_orange_pixels, orange_pixels)
    return result

def add_greenery(img_green, img_rest):
    green_hsv = cv2.cvtColor(img_green, cv2.COLOR_BGR2HSV)

    # Define the range of orange color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create a mask for pixels in the orange range
    green_mask = cv2.inRange(green_hsv, lower_green, upper_green)

    # Apply the mask to the "backtorgb_highway" image
    green_pixels = cv2.bitwise_and(img_green, img_green, mask=green_mask)

    # Resize the mask to match the dimensions of the "backtorgb_road" image
    resized_mask = cv2.resize(green_mask, (img_rest.shape[1], img_rest.shape[0]))

    # Create an inverted mask for the "backtorgb_road" image
    inv_mask = cv2.bitwise_not(resized_mask)

    # Extract the orange pixels from the "backtorgb_road" image
    road_orange_pixels = cv2.bitwise_and(img_rest, img_rest, mask=inv_mask)

    # Combine the extracted orange pixels and the original "backtorgb_road" image
    result = cv2.add(road_orange_pixels, green_pixels)
    return result

def get_blue(input):
    lower_blue = np.array([70, 50, 50])
    upper_blue = np.array([255, 255, 255])

    blue_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(blue_hsv, lower_blue, upper_blue)

    kernel_blue= np.ones((10, 10), np.uint8)
    closed_mask_blue = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_blue)
    backtorgb_blue = cv2.cvtColor(closed_mask_blue,cv2.COLOR_GRAY2RGB)

    backtorgb_blue[np.all(backtorgb_blue == (255, 255, 255), axis=-1)] = (255,0,0) #this is in bgr!
    return backtorgb_blue

def add_sea(img_blue, img_rest):
    blue_hsv = cv2.cvtColor(img_blue, cv2.COLOR_BGR2HSV)

    # Define the range of orange color in HSV
    lower_blue = np.array([70, 50, 50])
    upper_blue = np.array([255, 255, 255])

    # Create a mask for pixels in the orange range
    blue_mask = cv2.inRange(blue_hsv, lower_blue, upper_blue)

    # Apply the mask to the "backtorgb_highway" image
    green_pixels = cv2.bitwise_and(img_blue, img_blue, mask=blue_mask)

    # Resize the mask to match the dimensions of the "backtorgb_road" image
    resized_mask = cv2.resize(blue_mask, (img_rest.shape[1], img_rest.shape[0]))

    # Create an inverted mask for the "backtorgb_road" image
    inv_mask = cv2.bitwise_not(resized_mask)

    # Extract the orange pixels from the "backtorgb_road" image
    road_orange_pixels = cv2.bitwise_and(img_rest, img_rest, mask=inv_mask)

    # Combine the extracted orange pixels and the original "backtorgb_road" image
    result = cv2.add(road_orange_pixels, green_pixels)
    return result

def getgrey(input): #grey => red = urban
    lower_grey = np.array([0, 0, 30])
    upper_grey = np.array([255, 20, 240])
    grey_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    grey_mask = cv2.inRange(grey_hsv, lower_grey, upper_grey)

    kernel_red= np.ones((2, 2), np.uint8)
    closed_mask_grey = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, kernel_red)
    backtorgb_grey = cv2.cvtColor(closed_mask_grey,cv2.COLOR_GRAY2RGB)

    backtorgb_grey[np.all(backtorgb_grey == (255, 255, 255), axis=-1)] = (0,0,255) #this is in bgr!
    return backtorgb_grey

def add_grey(img_grey, img_rest): #grey => red = urban
    grey_hsv = cv2.cvtColor(img_grey, cv2.COLOR_BGR2HSV)

    # Define the range of grey color in HSV
    lower_grey = np.array([0, 240, 240])
    upper_grey = np.array([0, 255, 255])

    # Create a mask for pixels in the orange range
    grey_mask = cv2.inRange(grey_hsv, lower_grey, upper_grey)

    # Apply the mask to the "backtorgb_highway" image
    grey_pixels = cv2.bitwise_and(img_grey, img_grey, mask=grey_mask)

    # Resize the mask to match the dimensions of the "backtorgb_road" image
    resized_mask = cv2.resize(grey_mask, (img_rest.shape[1], img_rest.shape[0]))

    # Create an inverted mask for the "backtorgb_road" image
    inv_mask = cv2.bitwise_not(resized_mask)

    # Extract the orange pixels from the "backtorgb_road" image
    rest_pixels = cv2.bitwise_and(img_rest, img_rest, mask=inv_mask)

    # Combine the extracted orange pixels and the original "backtorgb_road" image
    result = cv2.add(rest_pixels, grey_pixels)
    return result

def getblack(input): #black should be buildings rest

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([0, 1, 0])

    black_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(black_hsv, lower_black, upper_black)

    kernel_black= np.ones((2, 2), np.uint8)
    closed_mask_black = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_black)
    backtorgb_black = cv2.cvtColor(closed_mask_black,cv2.COLOR_GRAY2RGB)

    backtorgb_black[np.all(backtorgb_black == (255, 255, 255), axis=-1)] = (255,255,0) #this is in bgr!
    return backtorgb_black

def add_black(img_black, img_rest): #grey => red = urban
    black_hsv = cv2.cvtColor(img_black, cv2.COLOR_BGR2HSV)
    
    lower_black = np.array([1, 1, 1])
    upper_black = np.array([255, 255, 255])

    # Create a mask for pixels in the orange range
    black_mask = cv2.inRange(black_hsv, lower_black, upper_black)

    # Apply the mask to the "backtorgb_highway" image
    black_pixels = cv2.bitwise_and(img_black, img_black, mask=black_mask)

    # Resize the mask to match the dimensions of the "backtorgb_road" image
    resized_mask = cv2.resize(black_mask, (img_rest.shape[1], img_rest.shape[0]))

    # Create an inverted mask for the "backtorgb_road" image
    inv_mask = cv2.bitwise_not(resized_mask)

    # Extract the orange pixels from the "backtorgb_road" image
    rest_pixels = cv2.bitwise_and(img_rest, img_rest, mask=inv_mask)

    # Combine the extracted orange pixels and the original "backtorgb_road" image
    result = cv2.add(rest_pixels, black_pixels)
    return result

def crop(input):
    height, width, channels = input.shape

    # Calculate the center of the image
    center_x = width // 2

    # Crop the right half of the image
    cropped_image = input[:, center_x:]
    return cropped_image

def addtobase(img1, img_changed):
    height, width = img1.shape[:2]
    start_col = width // 2
    img_changed = img_changed[:height, :start_col]
    img1[:, start_col:] = img_changed
    return img1
# Read the image

test = input("input the amount of images you wish to process")
test2 = input("input the starting index")
x = range(int(test)+1)
for n in x:
    index:int = int(test2)+n
    img = cv2.imread("maps\\maps\\train\\"+str(index)+".jpg")
    
    road = getroad(img)
    highway = gethighway(img)
    roadandhighway = combine_highway_and_road(highway, road)
    green = getgreen(img)
    addedgreen = add_greenery(green, roadandhighway)
    blue = get_blue(img)
    addedblue = add_sea(blue, addedgreen)
    houses = getgrey(img)
    addedhousing = add_grey(houses, addedblue)
    final = addtobase(img, crop(addedhousing)) #if i remove this it doesnt work? idk why
    buildings = getblack(img)
    addedbuildings = add_black(buildings, addedhousing)
    final2 = addtobase(img, crop(addedbuildings))
    #show
    #cv2.imshow('Base Image', cv2.imread("maps\\maps\\train\\"+str(index)+".jpg"))
    #cv2.imshow("Buildings", final2)

    # SAVING
    savingpath = "maps\\maps\\train_processed\\"+str(index)+".jpg"
    cv2.imwrite(savingpath, final2)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


#TODO
#BROWN ROADS CANT REALLY BE FOUND YET
#BLACK = URBAN?/HOUSING -> REST