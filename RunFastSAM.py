import numpy as np
import torch
from fastsam import FastSAM, FastSAMPrompt
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from scipy import ndimage
from scipy.ndimage import center_of_mass

model = FastSAM('FastSAM-s.pt')
IMAGE_PATH = "Le_Chocolat_Noir\WIN_20240131_14_50_38_Pro.jpg"
DEVICE = 'cpu'
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

# Display outlines of masks 10 to 18 on the original image
for i in range(10, 19):
    mask = sorted_masks[i]
    area = mask_areas[sorted_indices[i]].item()
    print(f"Area of mask {i}: {area}")

    # Create a new figure for this mask
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the original image
    ax.imshow(orig_img_float)

    # Find contours in the mask
    contours = find_contours(mask.cpu().numpy(), 0.5)

    # Display each contour
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    # Calculate the center of the mask
    # Calculate the center of the mask
    center = center_of_mass(mask.cpu().numpy())

    # Display the center of the mask
    ax.plot(center[1], center[0], 'ro')

    plt.show()