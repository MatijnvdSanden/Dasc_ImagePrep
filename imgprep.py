import cv2
import numpy as np
import argparse
from scipy.stats import mode

##
#For the love of god, dont change the colors it searches for in Imageprep, it will break everything.
##

class Imageprep():
    def __init__(self):
        self.parser()

    def parser(self):
        self.parser = argparse.ArgumentParser(description='How many images do you want to process? and the starting index of the images? and the type of images?')
        self.parser.add_argument('--amount', type=int, default=1, help='amount of images to process', required=True)
        self.parser.add_argument('--start', type=int, default=1, help='starting index', required=True)
        self.parser.add_argument('--type', type=str, default="train", help='train or val or tmp', required=True)
        self.parser.add_argument('--show', action=argparse.BooleanOptionalAction, help="save or show", default=True)

        self.args = self.parser.parse_args()

    def imgprep(self, img):
        road = self.getroad(img)
        highway = self.gethighway(img)
        roadandhighway = self.combine_highway_and_road(highway, road)
        green = self.getgreen(img)
        addedgreen = self.add_greenery(green, roadandhighway)
        blue = self.get_blue(img)
        addedblue = self.add_sea(blue, addedgreen)
        urban = self.getgrey(img)
        addedhousing = self.add_grey(urban, addedblue)
        final = self.addtobase(img, self.crop(addedhousing)) #if i remove this it doesnt work? idk why
        buildings = self.getblack(img)
        addedbuildings = self.add_black(buildings, addedhousing)
        final2 = self.addtobase(img, self.crop(addedbuildings))
        return final2

    def getroad(self, input):
        # Define the lower and upper bounds for the road color
        lower_road = np.array([241, 241, 241], dtype="uint8")
        upper_road = np.array([255, 255, 255], dtype="uint8")
        # Create a mask to extract the road color
        road_mask = cv2.inRange(input, lower_road, upper_road)


        #kernel_close = np.ones((4,4), np.uint8)
        #result_road = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_close, iterations=2)


        backtorgb_road = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2RGB)
        return backtorgb_road


    def gethighway(self, input):
        #Highway
        lower_highway = np.array([20, 70, 150], dtype="uint8")
        upper_highway = np.array([165, 255, 255], dtype="uint8")

        highway_mask = cv2.inRange(input, lower_highway, upper_highway)

        kernel_highway= np.ones((1, 1), np.uint8)
        closed_mask_highway = cv2.morphologyEx(highway_mask, cv2.MORPH_CLOSE, kernel_highway, iterations=2)
        
        # Remove small white dots in the black region
        kernel_open = np.ones((1, 1), np.uint8)
        closed = cv2.morphologyEx(closed_mask_highway, cv2.MORPH_OPEN, kernel_open, iterations=2)

        kernel_close = np.ones((1,1), np.uint8)
        result_highway = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        kernel_dilate = np.ones((2, 2), np.uint8)  # You can adjust the kernel size as needed
        expanded_highway = cv2.dilate(result_highway, kernel_dilate, iterations=4)

        backtorgb_highway = cv2.cvtColor(expanded_highway,cv2.COLOR_GRAY2RGB)

        backtorgb_highway[np.all(backtorgb_highway == (255, 255, 255), axis=-1)] = (0,255,255) #this is in bgr!
        return backtorgb_highway

    def getgreen(self, input):

        lower_green = np.array([30, 30, 30])
        upper_green = np.array([255, 255, 255])

        green_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(green_hsv, lower_green, upper_green)

        backtorgb_green = cv2.cvtColor(green_mask,cv2.COLOR_GRAY2RGB)

        backtorgb_green[np.all(backtorgb_green == (255, 255, 255), axis=-1)] = (0,255,0) #this is in bgr!
        return backtorgb_green

    def combine_highway_and_road(self, img_highway, img_road):
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

    def add_greenery(self, img_green, img_rest):
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

    def get_blue(self, input):
        lower_blue = np.array([70, 50, 50])
        upper_blue = np.array([255, 255, 255])

        blue_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(blue_hsv, lower_blue, upper_blue)

        backtorgb_blue = cv2.cvtColor(blue_mask,cv2.COLOR_GRAY2RGB)

        backtorgb_blue[np.all(backtorgb_blue == (255, 255, 255), axis=-1)] = (255,0,0) #this is in bgr!
        return backtorgb_blue

    def add_sea(self, img_blue, img_rest):
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

    def getgrey(self, input): #grey => red = urban
        lower_grey = np.array([0, 0, 1])
        upper_grey = np.array([255, 55, 238])
        grey_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        grey_mask = cv2.inRange(grey_hsv, lower_grey, upper_grey)

        #kernel_red= np.ones((2, 2), np.uint8)
        #closed_mask_grey = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, kernel_red)
        backtorgb_grey = cv2.cvtColor(grey_mask,cv2.COLOR_GRAY2RGB)    

        backtorgb_grey[np.all(backtorgb_grey == (255, 255, 255), axis=-1)] = (0,0,255) #this is in bgr!
        return backtorgb_grey

    def add_grey(self, img_grey, img_rest): #grey => red = urban
        grey_hsv = cv2.cvtColor(img_grey, cv2.COLOR_BGR2HSV)

        # Define the range of grey color in HSV
        lower_grey = np.array([0, 254, 254])
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

    def getblack(self, input): #black should be buildings rest

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([0, 1, 0])

        black_hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        black_mask = cv2.inRange(black_hsv, lower_black, upper_black)

        #kernel_black= np.ones((2, 2), np.uint8)
        #closed_mask_black = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_black)
        backtorgb_black = cv2.cvtColor(black_mask,cv2.COLOR_GRAY2RGB)

        backtorgb_black[np.all(backtorgb_black == (255, 255, 255), axis=-1)] = (255,255,0) #this is in bgr!
        return backtorgb_black

    def add_black(self, img_black, img_rest): #grey => red = urban
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

    def crop(self, input):
        height, width, channels = input.shape

        # Calculate the center of the image
        center_x = width // 2

        # Crop the right half of the image
        cropped_image = input[:, center_x:]
        return cropped_image

    def addtobase(self, img1, img_changed):
        height, width = img1.shape[:2]
        start_col = width // 2
        img_changed = img_changed[:height, :start_col]
        img1[:, start_col:] = img_changed
        return img1
    # Read the image

class Superpixelhandeling():
    def __init__(self):
        self.defined_colors_and_ranges = np.array([
        {'color': [172, 222, 204], 'range': 10, 'newcolor': [0,255,0]},  # Greenery
        {'color': [255, 255, 255], 'range': 10, 'newcolor': [255,255,255]},  # Road
        {'color': [240, 240, 240], 'range': 10, 'newcolor': [255,255,0]},  # Buildings
        {'color': [230, 230, 230], 'range': 10, 'newcolor': [0,0,255]},  # Urban
        {'color': [245, 155, 30], 'range': 10, 'newcolor': [0,255,255]}, #Highway
        {'color': [245, 220, 150], 'range': 10, 'newcolor': [245, 155, 30]}, #Highway2
        {'color': [175, 205, 245], 'range': 10, 'newcolor': [255,0,0]} # Water
        # Add more colors and ranges as needed
        ])
        self.fallback_color = np.array([230, 230, 230]) 


    def crop(self, input):
        height, width, channels = input.shape

        # Calculate the center of the image
        center_x = width // 2

        # Crop the right half of the image
        cropped_image = input[:, center_x:]
        return cropped_image

    def addtobase(self, img1, img_changed):
        height, width = img1.shape[:2]
        start_col = width // 2
        img_changed = img_changed[:height, :start_col]
        img1[:, start_col:] = img_changed
        return img1
    
    def dosuperpixels(self, inputimg):
        img = inputimg
        orig = self.crop(img)

        # Create a SuperpixelSLIC object
        algorithm = cv2.ximgproc.createSuperpixelSLIC(image=orig, algorithm=cv2.ximgproc.SLIC, region_size=10, ruler=25) #in mytesting SLIC > SLICO and best vars are now set!

        # Iterate to refine the superpixels
        algorithm.iterate()

        # Get the labels assigned to each pixel
        labels = algorithm.getLabels()
        greenery_index = next((index for index, color_info in enumerate(self.defined_colors_and_ranges) if np.all(color_info['color'] == [172, 222, 204])), None)
        white_index = next((index for index, color_info in enumerate(self.defined_colors_and_ranges) if np.all(color_info['color'] == [255, 255, 255])), None)
        highway_index = next((index for index, color_info in enumerate(self.defined_colors_and_ranges) if np.all(color_info['color'] == [245, 155, 30])), None)
        highway2_index = next((index for index, color_info in enumerate(self.defined_colors_and_ranges) if np.all(color_info['color'] == [245, 220, 150])), None)
        # Get the most common color in each superpixel that falls within the specified range
        superpixel_matched_color = {}
        for label in np.unique(labels):
            mask = (labels == label)
            superpixel_color = orig[mask]
            if np.any(np.all(np.abs(superpixel_color - self.defined_colors_and_ranges[greenery_index]['color']) <= self.defined_colors_and_ranges[greenery_index]['range'], axis=1)):
                # If greenery color is found, set the entire superpixel to greenery color                
                superpixel_matched_color[label] = self.defined_colors_and_ranges[greenery_index]['color']
            elif np.any(np.all(np.abs(superpixel_color - self.defined_colors_and_ranges[highway2_index]['color']) <= self.defined_colors_and_ranges[highway2_index]['range'], axis=1)):
                superpixel_matched_color[label] = self.defined_colors_and_ranges[highway_index]['color']
            else:
                # Calculate mean and mode colors
                mean_color = np.mean(superpixel_color, axis=0)
                mode_color, _ = mode(superpixel_color, axis=0, keepdims=True)

                # Check if the mean color falls within the specified range for any predefined color
                for defined_color in self.defined_colors_and_ranges:
                    color = np.array(defined_color['color'])
                    color_range = defined_color['range']

                    # Choose either mean or mode color based on your conditions
                    if np.all(np.abs(mean_color - color) <= color_range):
                        #superpixel_matched_color[label] = mean_color
                        if np.array_equal(color, [245, 220, 150]): #if its highway2 set it to highway
                            superpixel_matched_color[label] = [245, 155, 30]
                        else:
                            superpixel_matched_color[label] = defined_color['color']
                        break
                    elif np.all(np.abs(mode_color.squeeze() - color) <= color_range):
                        #superpixel_matched_color[label] = mode_color.squeeze()
                        if np.array_equal(color, [245, 220, 150]): #if its highway2 set it to highway
                            superpixel_matched_color[label] = [245, 155, 30]
                        else:
                            superpixel_matched_color[label] = defined_color['color']
                        break
                else:
                    # If no match is found for both mean and mode, use mean color
                    superpixel_matched_color[label] = mean_color

        # Optionally, you can visualize the superpixels
        output_image = algorithm.getLabelContourMask(thick_line=True)

        # Create a copy of the original image
        output_overlay = orig.copy()

        # Set the entire superpixel to its matched color on the original image
        for label, matched_color in superpixel_matched_color.items():
            output_overlay[labels == label] = matched_color

        out = self.addtobase(img.copy(), output_overlay)
        return out, output_image
    

if __name__ == "__main__":
    imgprep = Imageprep() #Init the image prepping class
    superpixelhandeling = Superpixelhandeling() #Init the superpixel handeling class
    print(f"starting at {imgprep.args.start} and processing {imgprep.args.amount} images") 
    for i in range(imgprep.args.start, imgprep.args.amount+1):
        imgloc = f"maps\\maps\\{imgprep.args.type}\\{i}.jpg" #location of the image
        orgimg = cv2.imread(imgloc) #read the image
        supimg = superpixelhandeling.dosuperpixels(orgimg) #apply superpixels to the image (preprocessing)
        img = imgprep.imgprep(supimg[0].copy()) #process the image 
        if (imgprep.args.show == True): #show or save the image
            cv2.imshow('superpixels', supimg[1])
            cv2.imshow('superpixels image', supimg[0])
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            savingpath = f"maps\\maps\\{imgprep.args.type}_superpixels_2_processed\\{i}.jpg"
            cv2.imwrite(savingpath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print(f"saved at :{savingpath}")
