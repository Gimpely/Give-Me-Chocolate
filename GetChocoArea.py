import numpy as np
import torch
from fastsam import FastSAM, FastSAMPrompt
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from scipy import ndimage
from scipy.ndimage import center_of_mass
from skimage.measure import label, regionprops


############################################################
model = FastSAM('FastSAM-s.pt')
IMAGE_PATH = "Cokolada\WIN_20240131_14_50_32_Pro.jpg"
DEVICE = 'cpu'
#desired chocolate area range
max_area = 44000
min_area = 30000
############################################################




everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

ann = prompt_process.everything_prompt()
mask_data = everything_results[0].masks.data

mask_areas = torch.sum(mask_data, dim=(1,2))

# Sort the masks and their areas in descending order of area
sorted_indices = torch.argsort(mask_areas, descending=True)
sorted_masks = mask_data[sorted_indices]

# Get the original image
orig_img = everything_results[0].orig_img

# Convert the original image to float
orig_img_float = orig_img.astype(float) / 255

# Display outlines of masks with an area between 43000 and 44000 on the original image
for i in range(len(sorted_masks)):
    mask = sorted_masks[i]
    area = mask_areas[sorted_indices[i]].item()

    # Check if the area of the mask is between 43000 and 44000
    if min_area < area < max_area:
        mask_name = f"Area of mask {i}: {area}"
        print(mask_name)

        # Create a new figure for this mask
        plt.figure(figsize=(10, 10))

        # Add a title to the figure
        plt.title(mask_name)

        # Display the original image
        plt.imshow(orig_img_float)

        # Label all connected regions in the mask
        labels = label(mask.cpu().numpy())

        # Keep only the largest region
        largest_region = max(regionprops(labels), key=lambda x: x.area)
        mask_largest = labels == largest_region.label

        # Find contours in the largest mask
        contours = find_contours(mask_largest, 0.5)

        # Display each contour
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)

        # Calculate the center of the mask
        center = center_of_mass(mask_largest)

        # Display the center of the mask
        plt.plot(center[1], center[0], 'ro')

# Show all figures
plt.show()