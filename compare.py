import cv2

def compare_images(image1_path, image2_path):
    # Load the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Get the width of the images
    width = image1.shape[1]

    # Calculate the midpoint
    midpoint = width // 2

    # Cut off the left part of the images
    image1_cropped = image1[:, midpoint:]
    image2_cropped = image2[:, midpoint:]

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1_cropped, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2_cropped, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the images
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference image to highlight the changes
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding rectangles around the contours
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image1_cropped, (x, y), (x + w, y + h), (255, 0, 255), 1)
        cv2.rectangle(image2_cropped, (x, y), (x + w, y + h), (255, 0, 255), 1)

    # Display the cropped images with differences highlighted
    cv2.imshow("Cropped Image 1", image1_cropped)
    cv2.imshow("Cropped Image 2", image2_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
#image1_path = "maps\\maps\\train_superpixels_processed\\10.jpg"
image1_path = "maps\\maps\\train_processed\\10.jpg"
image2_path = "maps\\maps\\train_superpixels_2_processed\\10.jpg"
compare_images(image1_path, image2_path)