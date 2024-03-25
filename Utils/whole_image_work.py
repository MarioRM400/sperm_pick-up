# %% imports 
import yolov5 
import cv2
import pandas
import torch 
import os
import numpy as np
import math
import segmentation_models_pytorch as smp
import torchvision.transforms as tvt

project_dir = r"C:\Users\PC KAIJU\ConceivableProjectsTools\SPERM-TAIL-DET-SEG\sperm_pick-up"
det_weights_path = os.path.join(project_dir, "weights", "sperm.pt")
seg_weights_path = os.path.join(project_dir, "weights", "unet.pt")

CONF_THR = 0.55
IOU_THR = 0.45
N_CLASSES = 3
N_CHANNELS = 1 
HEIGHT = 384  # image height
WIDTH = 448  # image width
MAX_DET = 10
cut_size = (WIDTH, HEIGHT) # Specify your target size

classes = ["head", "needle_tip", "tail"]

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# image stuff
image = os.path.join(project_dir, "Data", "sperm1.jpg")
image = cv2.imread(image)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gs_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Get the dimensions of the image
height, width, _ = img.shape
# Calculate the central point coordinates
center_x = int(width // 2)  
center_y = int(height // 2)
image_centroid = (center_x, center_y)

# %% FUNCTION LOAD
def load_det_model(det_weights_path):
    model = torch.hub.load('ultralytics/yolov5', 
                        'custom', 
                        path=det_weights_path, 
                        force_reload=True) 

    det_model = model.to(DEVICE)
    det_model.compute_iou = IOU_THR  # IoU threshold
    det_model.conf = CONF_THR # Confidence threshold
    det_model.max_det = MAX_DET  # Max number of detections
    return det_model

# Seg-model stuff
def load_seg_model(seg_weights_path):
    seg_model = smp.Unet(
                encoder_name="resnet101",
                encoder_weights='imagenet',
                in_channels=N_CHANNELS,
                classes=N_CLASSES
                )
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Puts the model in the device (CPU or GPU)
    seg_model = seg_model.to(DEVICE)
    # Loads the trained weights
    seg_model.load_state_dict(torch.load(seg_weights_path))
    # Sets the segmentation model to evaluation mode
    seg_model.eval()
    return seg_model


def crop_sperm(bbxs):
    sperms_info = []
    sperm_centroids = []
    distances = []
    for bbx in range(len(bbxs)):
        x_min, y_min, x_max, y_max, conf, class_n, class_name = bbxs[bbx]  
        if class_name == 'head':
            # cropped_image = img[int(y_min):int(y_max), int(x_min):int(x_max)]
            # cv2.imwrite(os.path.join(project_dir, 
            #                          f'sperm{bbx + 1}.jpg'), cropped_image)
            bbx_x_center = (x_min + x_max) / 2
            bbx_y_center = (y_min + y_max) / 2
            sperms_info.append(bbxs[bbx])
            sperm_centroids.append((int(bbx_x_center), int(bbx_y_center)))
            distance = math.sqrt((bbx_x_center - center_x)**2 + (bbx_y_center - center_y)**2)
            distances.append(int(distance))
            marker_color = (0, 255, 0)  # Green color (BGR format)

    closest_sperm = distances.index(min(distances))

    x_min, y_min, x_max, y_max, conf, class_n, class_name = sperms_info[closest_sperm] 

    cropped_image = gs_img[int(y_min):int(y_max), int(x_min):int(x_max)]
    cv2.imwrite(os.path.join(project_dir, 
                            f'sperm{closest_sperm}.jpg'), cropped_image)
    
    return x_min, y_min, x_max, y_max, cropped_image


def pad_image(image, target_size):
    """
    Pad image to target size.
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate padding
    pad_width = max(0, target_width - width)
    pad_height = max(0, target_height - height)

    # Calculate padding amounts for left, top, right, bottom
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding
    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    # save image
    padded_image_name = os.path.join(project_dir, 
                          'padded_sperm.jpg')

    cv2.imwrite(padded_image_name, padded_image)

    return padded_image


def seg_sperm(padded_image, seg_model):
    transform_img = tvt.Compose([tvt.ToPILImage(),
                                    tvt.Resize((HEIGHT, WIDTH), interpolation=tvt.InterpolationMode.BILINEAR),
                                    tvt.ToTensor()])

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    padded_image_ = clahe.apply(padded_image).astype(np.float32)
    padded_image_ = transform_img(padded_image_)
    padded_image_ = torch.autograd.Variable(padded_image_, requires_grad=False).to(DEVICE).unsqueeze(0)


    with torch.no_grad():
                # Gets the prediction
                prd = seg_model(padded_image_)

    prd = (torch.sigmoid(prd) > 0.5).long()
    prd = tvt.Resize((HEIGHT, WIDTH), tvt.InterpolationMode.NEAREST)(prd[0])

    seg = prd.data.cpu().detach().numpy()
    seg = np.swapaxes(seg,0,2)
    seg = np.swapaxes(seg, 1, 0)
    segmented_sperm = (seg * 255).astype(np.uint8)

    seg_mask = os.path.join(project_dir, 
                            "seg.jpg")

    cv2.imwrite(seg_mask, segmented_sperm)

    return segmented_sperm


def crop_image(padded_image, padded_mask, cropped_image):
    """
    Crop padded image back to original size.
    """
    padded_height, padded_width = padded_image.shape[:2]
    original_height, original_width = cropped_image.shape

    # Calculate cropping dimensions
    crop_top = (padded_height - original_height) // 2
    crop_bottom = crop_top + original_height
    crop_left = (padded_width - original_width) // 2
    crop_right = crop_left + original_width

    # Perform cropping
    cropped_image = padded_image[crop_top:crop_bottom, crop_left:crop_right]
    cropped_mask = padded_mask[crop_top:crop_bottom, crop_left:crop_right]

    rev_mask = os.path.join(project_dir, 
                          "reverted_padd_mask.jpg")
    cv2.imwrite(rev_mask, cropped_mask)

    return cropped_image, cropped_mask





def get_shoot_coords(x_min, y_min, resized_mask):
    # rsized image centroid
    r_height, r_width, _ = resized_mask.shape
    # Calculate the central point coordinates
    r_center_x = int(r_width // 2)
    r_center_y = int(r_height // 2)
    # resized mask centroid 
    r_coord = (int(r_center_x), int(r_center_y))

    # gray_mask = cv2.cvtColor(resized_mask[:,:,2], cv2.COLOR_BGR2GRAY)
    # # Apply thresholding to create a binary image
    _, thresholded_image = cv2.threshold(resized_mask[:,:,2], 0, 255, cv2.THRESH_BINARY)

    # # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # # Draw the largest contour on the original image
    cv2.drawContours(resized_image, [largest_contour], -1, (0, 255, 0), 2)

    coordinates = []
    r_distances = []
    for point in largest_contour:
        x, y = point[0]
        coordinates.append((x, y))    
        r_distance = math.sqrt((x - r_center_x)**2 + (y - r_center_y)**2)
        r_distances.append(int(r_distance))
        
    closest_r_coord = r_distances.index(min(r_distances)) 
    shoot_coordinate = coordinates[closest_r_coord]

    shoot_coordinate = (int(x_min + shoot_coordinate[0]),
                        int(y_min + shoot_coordinate[1]))
    

    return r_coord, shoot_coordinate, largest_contour


def polygon_centroid(vertices):
    n = len(vertices)
    A = 0
    Cx = 0
    Cy = 0

    for i in range(n):
        xi, yi = vertices[i]
        xi_plus_1, yi_plus_1 = vertices[(i + 1) % n]

        # Update the area
        A += (xi * yi_plus_1 - xi_plus_1 * yi)

        # Update the centroid coordinates
        Cx += (xi + xi_plus_1) * (xi * yi_plus_1 - xi_plus_1 * yi)
        Cy += (yi + yi_plus_1) * (xi * yi_plus_1 - xi_plus_1 * yi)

    A *= 0.5
    Cx /= (6 * A)
    Cy /= (6 * A)

    return (int(Cx), int(Cy))


def get_pickup_coordinate(x_min, y_min, resized_mask, tail_coordinates):
    _, thresholded_head_image = cv2.threshold(resized_mask[:,:,1], 0, 255, cv2.THRESH_BINARY)
    
    # # Find contours in the binary image
    head_contours, _ = cv2.findContours(thresholded_head_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # Find the contour with the largest area
    largest_head_contour = min(head_contours, key=cv2.contourArea)
            
    head_coordinates = []
    # hed_r_distances = []
    for point in largest_head_contour:
        x, y = point[0]
        head_coordinates.append((x, y))    
        # head_r_distance = math.sqrt((x - head_centroid[0])**2 + (y - head_centroid[1])**2)
        # hed_r_distances.append(int(head_r_distance))
        
    head_centroid = polygon_centroid(head_coordinates)
    
    tail_coordinates_un = []
    t_distances = []
    for point in tail_coordinates:
        x, y = point[0]
        tail_coordinates_un.append((x, y))
        t_distance = math.sqrt((x - head_centroid[0])**2 + (y - head_centroid[1])**2)
        t_distances.append(int(t_distance))
    
    farest_t_coord = t_distances.index(max(t_distances)) 
    farest_t_coord = tail_coordinates_un[farest_t_coord]
    
    pick_up_coordinate = (int(x_min + farest_t_coord[0]),
                        int(y_min + farest_t_coord[1]))
    
    head_centroid = (int(x_min + head_centroid[0]),
                        int(y_min + head_centroid[1]))
    return pick_up_coordinate, head_centroid

# %% detection and model results 
det_model = load_det_model(det_weights_path)
seg_model = load_seg_model(seg_weights_path)

detections = det_model(gs_img)
bbxs = detections.pandas().xyxy[0]
bbxs = bbxs.values.tolist()
# %% BBx definition
x_min, y_min, x_max, y_max, cropped_image = crop_sperm(bbxs)
# %% Pad cropped image to a target size
padded_image = pad_image(cropped_image, cut_size)

# %% Segmenter process initialize
segmented_sperm = seg_sperm(padded_image, seg_model)
# %% unpadd image 
resized_image, resized_mask = crop_image(padded_image, segmented_sperm, cropped_image)

bbx_centroid, shoot_coordinate, tail_coordinates = get_shoot_coords(x_min, y_min, resized_mask)

# %% image Centroid
# # Define the color of the point (in BGR format)
point_color = (0, 0, 255)  # Red color
# # # Define the radius of the point
radius = 5

cnt = img
# top_left_bbx_coord = (int(x_min), int(y_min))
# below_right_coord = (int(x_max), int(y_max))

cv2.circle(cnt, shoot_coordinate, radius, (0, 0, 255), -1)

# %% get the pick_up_coordinate
pick_up_coordinate, head_centroid = get_pickup_coordinate(x_min, y_min, resized_mask, tail_coordinates)

cv2.circle(cnt, head_centroid, radius, (0, 0, 255), -1)

point_color = (255, 255, 255) 
radius = 5
cv2.circle(cnt, pick_up_coordinate, radius, point_color, -1)
cv2.circle(cnt, head_centroid, radius, point_color, -1)
cv2.imshow("Image with Center Marker and Bounding Box", cnt)
cv2.waitKey(0)
cv2.destroyAllWindows()
