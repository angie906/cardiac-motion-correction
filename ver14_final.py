# ver 14 adaptive reference registration - o (final version)
import os
import numpy as np
import cv2
from tifffile import imwrite, imread
from scipy import ndimage
import matplotlib.pyplot as plt
import optimap as om

num = "907"

# =================================================
# Configuration
# =================================================
input_dir = f"Experiment_36_o/{num}"
output_dir = f"output_Farneback_fulloutput_Farneback_ver14_{num}"

os.makedirs(output_dir, exist_ok=True)

# adaptive reference parameter 
alpha = 0.95

# =================================================
# Load TIFF stack
# =================================================
def load_tiff_folder(folder):

    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith((".tif",".tiff"))])

    first = imread(os.path.join(folder,files[0]))

    n = len(files)
    h,w = first.shape

    video = np.zeros((n,h,w),dtype=np.float32)

    print(f"Loading {n} frames ({h}x{w})")

    for i,f in enumerate(files):
        video[i] = imread(os.path.join(folder,f)).astype(np.float32)

    print("Loading complete.")
    return video


video = load_tiff_folder(input_dir)

num_frames,h,w = video.shape

processed_video = np.zeros_like(video)

# =================================================
# Optical Flow Registration
# =================================================
def register_slice(base,target,inverse_map=True):

    base_blur = cv2.GaussianBlur(base,(3,3),1.0)
    target_blur = cv2.GaussianBlur(target,(3,3),1.0)

    flow = cv2.calcOpticalFlowFarneback(
        base_blur,
        target_blur,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=10,
        poly_n=5,
        poly_sigma=1.1,
        flags=0
    )

    grid_x,grid_y = np.meshgrid(np.arange(w),np.arange(h))

    if inverse_map:
        map_x = (grid_x + flow[...,0]).astype(np.float32)
        map_y = (grid_y + flow[...,1]).astype(np.float32)
    else:
        map_x = (grid_x - flow[...,0]).astype(np.float32)
        map_y = (grid_y - flow[...,1]).astype(np.float32)

    registered = cv2.remap(
        target,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR
    )

    return registered,flow


# =================================================
# Contrast Enhancement
# =================================================
def apply_contrast(frame):

    enhanced = om.motion.contrast_enhancement(frame,19)

    return cv2.normalize(
        enhanced,None,0,255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)


# =================================================
# Watershed Mask Creation
# =================================================
def create_watershed_mask(enhanced_uint8, min_area=500, dist_thresh_factor=0.2):

    blurred = cv2.GaussianBlur(enhanced_uint8,(5,5),0)

    # otsu thresholding
    _, thresh = cv2.threshold(
        blurred,0,255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # small elliptical kernel for morphological operations
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    # for removing small artifacts
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kern,iterations=2)

    # expanding clean region
    sure_bg = cv2.dilate(opening,kern,iterations=3)

    # distance from each pixel to nearest background
    dist = cv2.distanceTransform(opening,cv2.DIST_L2,5)

    # threshold the distance map to keep only certain foreground 
    _, sure_fg = cv2.threshold(
        dist,
        dist_thresh_factor * dist.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)
    
    # region close to boundaries
    near_border = cv2.subtract(sure_bg,sure_fg)

    # label connected components in the foreground
    num_labels,markers = cv2.connectedComponents(sure_fg)

    # background not zero 
    markers = markers + 1
    markers[near_border==255] = 0

    # grayscale image to color (watershed requires 3-channel image)
    img_color = cv2.cvtColor(enhanced_uint8,cv2.COLOR_GRAY2BGR)

    markers = cv2.watershed(img_color,markers.astype(np.int32))

    # final mask (keep segmented, exclude boundaries)
    mask = (markers>1).astype(np.uint8)*255

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1
    best_contour = None

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt,True)

        if perimeter == 0:
            continue

        
        circularity = 4*np.pi*area/(perimeter**2)
        # combine size and shape into one score (choose large, convex regions)
        score = area*circularity

        if score > best_score:
            best_score = score
            best_contour = cnt

    mask_clean = np.zeros_like(mask)

    if best_contour is not None:
        cv2.drawContours(mask_clean,[best_contour],-1,255,cv2.FILLED)
    
    # fill holes inside the region 
    mask_filled = ndimage.binary_fill_holes(mask_clean>0).astype(np.uint8)*255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

    smoothed = cv2.morphologyEx(mask_filled,cv2.MORPH_CLOSE,kernel)

    # fill again in case smoothing created small gaps
    smoothed = ndimage.binary_fill_holes(smoothed>0).astype(np.uint8)*255

    return smoothed


# =================================================
# Build Mask
# =================================================
max_proj = np.max(video,axis=0)

enhanced = apply_contrast(max_proj)

mask_uint8 = create_watershed_mask(enhanced)

mask_bool = mask_uint8.astype(bool)
mask_float = mask_bool.astype(np.float32)

plt.figure(figsize=(6,6))
plt.imshow(mask_uint8,cmap='gray')
plt.title("Watershed Mask")
plt.axis("off")
plt.show()

# =================================================
# store Motion displacement 
# =================================================
original_motion = []
residual_motion = []

# =================================================
# Reference frame 
# =================================================
reference_frame = video[0].astype(np.float32)

processed_video[0] = reference_frame * mask_float

stabilized_video = np.zeros_like(video)
stabilized_video[0] = processed_video[0]

# =================================================
# Main Loop
# =================================================
for i in range(1,num_frames):

    current = video[i].astype(np.float32)

    # registration
    stabilized,flow = register_slice(
        reference_frame,
        current,
        inverse_map=True
    )

    # -----------------------------
    # Original motion (Raw)
    # -----------------------------
    flow_mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)

    original_motion.append(
        float(np.mean(flow_mag[mask_bool]))
    )

    # -----------------------------
    # Residual motion (Corrected)
    # -----------------------------
    _, residual_flow = register_slice(
        reference_frame,
        stabilized,
        inverse_map=True
    )

    residual_mag = np.sqrt(
        residual_flow[...,0]**2 +
        residual_flow[...,1]**2
    )

    residual_motion.append(
        float(np.mean(residual_mag[mask_bool]))
    )

    stabilized_masked = stabilized * mask_float

    processed_video[i] = stabilized_masked
    stabilized_video[i] = stabilized_masked

    # adaptive reference update
    reference_frame = alpha * reference_frame + (1 - alpha) * stabilized

# =================================================
# Save corrected TIFF frames
# =================================================
print("\nSaving corrected TIFF frames...")

for i, frame in enumerate(processed_video):

    save_path = os.path.join(output_dir, f"frame_{i:04d}.tiff")

    imwrite(
        save_path,
        np.clip(frame, 0, 65535).astype(np.float32)
    )

print("All frames saved.")
# =================================================
# Motion Displacement Plot
# =================================================

fps = 400

t = np.arange(len(original_motion)) / fps

window = 5

smooth_original = np.convolve(
    original_motion,
    np.ones(window)/window,
    mode='valid'
)

smooth_residual = np.convolve(
    residual_motion,
    np.ones(window)/window,
    mode='valid'
)

t_smooth = t[window-1:]

plt.figure(figsize=(12,6))

plt.plot(
    t,
    original_motion,
    color='red',
    alpha=0.3,
    label="Raw Motion"
)

plt.plot(
    t,
    residual_motion,
    color='blue',
    alpha=0.3,
    label="Residual Motion"
)


plt.xlabel("Time (s)")
plt.ylabel("Average Pixel Displacement (px)")
plt.title("Motion Quantification: Original vs Motion-corrected")

plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(output_dir,f"motion_displacement_{num}_ver14.png")
)

plt.show()

# # =================================================
# # Save stacked TIFF (video)
# # =================================================
# imwrite(
#     os.path.join(output_dir,f"stacked_output_ver14_{num}.tiff"),
#     processed_video.astype(np.float32)
# )

print("\nDone. Adaptive reference output saved.")