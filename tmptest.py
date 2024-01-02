# import the necessary packages
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
from scipy.stats import mode
import argparse
import cv2

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

def find_nearest_color(target_color, color_list, tolerance=10):
    for color in color_list:
        if np.all(np.abs(target_color - color) <= tolerance):
            return color
    return None

img = cv2.imread('maps\\maps\\tmp\\1.jpg')
orig = crop(img)

# Define a set of colors and their ranges you want to look for (in BGR format)
defined_colors_and_ranges = np.array([
    {'color': [172, 222, 204], 'range': 10},  # Greenery
    {'color': [255, 255, 255], 'range': 10},  # Road
    {'color': [240, 240, 240], 'range': 10},  # Buildings
    {'color': [230, 230, 230], 'range': 10},  # Urban
    # Add more colors and ranges as needed
])

# Pink color (255, 192, 203) as a fallback
pink_color = np.array([203, 192, 255])

# Create a SuperpixelSLIC object
algorithm = cv2.ximgproc.createSuperpixelSLIC(image=orig, algorithm=cv2.ximgproc.SLICO, region_size=5)

# Iterate to refine the superpixels
algorithm.iterate()

# Get the labels assigned to each pixel
labels = algorithm.getLabels()

# Set the entire superpixel to the greenery color if any pixel within that superpixel matches the greenery color
output_overlay = orig.copy()

for label in np.unique(labels):
    mask = (labels == label)
    superpixel_color = orig[mask]
    mode_color, _ = mode(superpixel_color, axis=0)

    # Check if any pixel in the superpixel matches the greenery color
    greenery_color = find_nearest_color(mode_color.squeeze(), defined_colors_and_ranges[:, 'color'], tolerance=10)
    
    if greenery_color is not None and np.all(np.abs(mode_color.squeeze() - greenery_color) <= 10):
        output_overlay[labels == label] = greenery_color
    else:
        # If no match is found, use pink color
        output_overlay[labels == label] = pink_color

# Optionally, you can visualize the superpixels
output_image = algorithm.getLabelContourMask(thick_line=True)

out = addtobase(img, output_overlay)

# Display the original image, superpixel segmentation, and overlay with matched colors
show = True
if not show:
    cv2.imshow('Original Image', img)
    cv2.imshow('Superpixel Segmentation', output_image)
    cv2.imshow('Entire Superpixel with Matched Color', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    savingpath = "maps\\maps\\tmp\\out1.jpg"
    cv2.imwrite(savingpath, out)
