import numpy as np
import torch
from fastsam import FastSAM, FastSAMPrompt
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from scipy import ndimage
from scipy.ndimage import center_of_mass
from skimage.measure import label, regionprops
import os

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

model = FastSAM('FastSAM-s.pt')
DEVICE = 'cpu'

# Get a list of all files in the directory
image_files = os.listdir("Cokolada")

# Process only the first 5 images
for image_file in image_files[:5]:
    IMAGE_PATH = os.path.join("Cokolada", image_file)

    everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=512, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

    ann = prompt_process.everything_prompt()
    mask_data = everything_results[0].masks.data

    # Get the original image
    orig_img = everything_results[0].orig_img

    # Convert the original image to float
    orig_img_float = orig_img.astype(float) / 255

    # Create a new figure for this image
    plt.figure(figsize=(10, 10))

    # Display the original image
    plt.imshow(orig_img_float)

    # Display outlines of masks with an area between 43000 and 44000 on the original image
    for i, mask in enumerate(mask_data):
        area = torch.sum(mask).item()

        # Check if the area of the mask is between 43000 and 44000
        if 30000 < area < 50000:
            # Label all connected regions in the mask
            labels = label(mask.cpu().numpy())

            # Keep only the largest region
            largest_region = max(regionprops(labels), key=lambda x: x.area)
            mask_largest = labels == largest_region.label

            # Check for overlap with previous masks
            for j in range(i):
                prev_mask = mask_data[j]
                iou = calculate_iou(mask_largest, prev_mask.cpu().numpy())
                if iou > 0.5:
                    break
            else:
                # Get the bounding box of the largest region
                minr, minc, maxr, maxc = largest_region.bbox

                # Calculate the aspect ratio
                aspect_ratio = (maxc - minc) / (maxr - minr)

                # Check if the shape is mostly rectangular
                is_rectangular = 0.5 < aspect_ratio < 2

                # Check the solidity of the region
                solidity = largest_region.solidity

                # Consider the shape as rectangular if the aspect ratio is between 0.5 and 2 and the solidity is above 0.9
                is_rectangular = is_rectangular and solidity > 0.9

                # Print the area and whether the shape is mostly rectangular
                mask_name = f"Area of mask {i}: {area}, Rectangular: {'Yes' if is_rectangular else 'No'}"
                print(mask_name)

                # If the shape is rectangular, display it
                if is_rectangular:
                    # Find contours in the largest mask
                    contours = find_contours(mask_largest, 0.5)

                    # Display each contour
                    for contour in contours:
                        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

                    # Calculate the center of the mask
                    center = center_of_mass(mask_largest)

                    # Display the center of the mask
                    plt.plot(center[1], center[0], 'ro')

    # Show the figure for this image
    plt.show()