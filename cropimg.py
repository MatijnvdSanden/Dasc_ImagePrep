from PIL import Image

#in range of 1k images
for i in range(1, 1001):
    # Open the image

    loc = "val" #val or train

    openinglocation = "maps\\maps\\"+loc+f"_superpixels_processed\\{i}.jpg"
    savinglocation = "maps\\maps\\"+loc+f"_superpixels_processed_cropped\\{i}.jpg"

    print(f"image {i} being processed")

    image = Image.open(openinglocation)

    # Crop the left image
    left_image = image.crop((0, 0, 600, 600))
    left_image = left_image.resize((512, 512))

    # Crop the right image
    right_image = image.crop((600, 0, 1200, 600))
    right_image = right_image.resize((512, 512))

    # Create a new image with the final size
    final_image = Image.new("RGB", (1024, 512))

    # Paste the left image on the left side of the final image
    final_image.paste(left_image, (0, 0))

    # Paste the right image on the right side of the final image
    final_image.paste(right_image, (512, 0))

    # Display the final image

    final_image.save(savinglocation)