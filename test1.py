# import the necessary packages
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
from scipy.stats import mode
import argparse
import cv2

'''
TODO
ADD HIGHWAY AND OCEAN TO THE COLORS IT SEARCHES FOR!
THIS SHOULD BE GOOD!
'''


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


parser = argparse.ArgumentParser(description='How many images do you want to process? and the starting index of the images? and the type of images?')
parser.add_argument('--amount', type=int, default=1, help='amount of images to process', required=True)
parser.add_argument('--start', type=int, default=1, help='starting index', required=True)
parser.add_argument('--type', type=str, default="train", help='train or val or tmp', required=True)
parser.add_argument('--save', action=argparse.BooleanOptionalAction, help="save or show", default=True)
args = parser.parse_args()


# Define a set of colors and their ranges you want to look for (in BGR format)
defined_colors_and_ranges = np.array([
    {'color': [172, 222, 204], 'range': 10, 'newcolor': [0,255,0]},  # Greenery
    {'color': [255, 255, 255], 'range': 10, 'newcolor': [255,255,255]},  # Road
    {'color': [240, 240, 240], 'range': 10, 'newcolor': [255,255,0]},  # Buildings
    {'color': [230, 230, 230], 'range': 10, 'newcolor': [0,0,255]},  # Urban
    {'color': [245, 155, 30], 'range': 10, 'newcolor': [0,255,255]}, #    Highway
    {'color': [175, 205, 245], 'range': 10, 'newcolor': [255,0,0]} # Water
    # Add more colors and ranges as needed
])

# Urban color (230, 230, 230) as a fallback
fallback_color = np.array([230, 230, 230]) 

amount = args.amount
start = args.start
typerunning = args.type
save = args.save
x = range(int(amount))
for n in x:
    index:int = int(start)+n
    imgloc = "maps\\maps\\" + typerunning + "\\"+str(index)+".jpg"
    img = cv2.imread(imgloc)
    orig = crop(img)

    # Create a SuperpixelSLIC object
    algorithm = cv2.ximgproc.createSuperpixelSLIC(image=orig, algorithm=cv2.ximgproc.SLIC, region_size=10, ruler=25) #in mytesting SLIC > SLICO and best vars are now set!

    # Iterate to refine the superpixels
    algorithm.iterate()

    # Get the labels assigned to each pixel
    labels = algorithm.getLabels()
    greenery_index = next((index for index, color_info in enumerate(defined_colors_and_ranges) if np.all(color_info['color'] == [172, 222, 204])), None)
    white_index = next((index for index, color_info in enumerate(defined_colors_and_ranges) if np.all(color_info['color'] == [255, 255, 255])), None)

    # Get the most common color in each superpixel that falls within the specified range
    superpixel_matched_color = {}
    for label in np.unique(labels):
        mask = (labels == label)
        superpixel_color = orig[mask]
        if np.any(np.all(np.abs(superpixel_color - defined_colors_and_ranges[greenery_index]['color']) <= defined_colors_and_ranges[greenery_index]['range'], axis=1)):
            # If greenery color is found, set the entire superpixel to greenery color
            superpixel_matched_color[label] = defined_colors_and_ranges[greenery_index]['color']
        else:
            # Calculate mean and mode colors
            mean_color = np.mean(superpixel_color, axis=0)
            mode_color, _ = mode(superpixel_color, axis=0)


            # Check if the mean color falls within the specified range for any predefined color
            closestrange = 0
            ownerofcloses = None
            for defined_color in defined_colors_and_ranges:
                color = np.array(defined_color['color'])
                color_range = defined_color['range']

                # Choose either mean or mode color based on your conditions
                if np.all(np.abs(mean_color - color) <= color_range):
                    #superpixel_matched_color[label] = mean_color
                    superpixel_matched_color[label] = defined_color['color']
                    break
                elif np.all(np.abs(mode_color.squeeze() - color) <= color_range):
                    #superpixel_matched_color[label] = mode_color.squeeze()
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

    out = addtobase(img.copy(), output_overlay)
    # Display the original image, superpixel segmentation, and overlay with matched colors
    if not save:
        cv2.imshow('Original Image', img)
        cv2.imshow('Superpixel Segmentation', output_image)
        cv2.imshow('Entire Superpixel with Matched Color', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("maps\\maps\\" +typerunning+ "_superpixels" + "\\"+str(index)+".jpg")
        savingpath = "maps\\maps\\" +typerunning+ "_superpixels" + "\\"+str(index)+".jpg"
        cv2.imwrite(savingpath, out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


