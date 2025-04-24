from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

import numpy as np
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import shutil


np.random.seed(42)
tf.random.set_seed(42)

with open('characters_list.pkl', 'rb') as file:
    ff_loaded = pickle.load(file)


AUTOTUNE = tf.data.AUTOTUNE

#mapping characters to integers
char_to_num = StringLookup(vocabulary = ff_loaded,mask_token = None)

#mapping integers back to original characters
num_to_char = StringLookup(
    vocabulary = char_to_num.get_vocabulary(),mask_token=None, invert = True
)

batch_size = 64
padding_token = 99
image_width = 128
image_height = 32

max_len = 27

class CTCLayer(keras.layers.Layer):
  def __init__(self,name=None):
    super().__init__(name=name)
    self.loss_fn = keras.backend.ctc_batch_cost

  def call(self,y_true,y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0],dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1],dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1],dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len,1),dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len,1),dtype="int64")
    loss = self.loss_fn(y_true,y_pred,input_length,label_length)
    self.add_loss(loss)

    return y_pred

yolo_model_path='yolo_models/best60epoch.pt'
model = YOLO(yolo_model_path)

custom_objects = {'CTCLayer': CTCLayer}
crnn_model_path = 'handwriting_recognizer50.h5'
with keras.utils.custom_object_scope(custom_objects):
    model = keras.models.load_model(crnn_model_path)

prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name = "dense2").output
)



def compute_area(box):
    """
    Computes the area of a bounding box.
    
    box: A tuple or list in the format (x1, y1, x2, y2) representing a bounding box.
    """
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def compute_containment_ratio(boxA, boxB):
    """
    Computes the containment ratio of boxA inside boxB.
    Returns a value between 0 and 1 representing the percentage of boxA that is contained within boxB.
    
    boxA, boxB: Tuples or lists in the format (x1, y1, x2, y2) representing bounding boxes.
    """
    # Coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of the intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Compute the area of the smaller box (boxA)
    boxAArea = compute_area(boxA)
    boxBArea = compute_area(boxB)

    # Return the containment ratio (percentage of small box inside big box)
    smaller_area = min(boxAArea,boxBArea)
    containment_ratio = interArea / float(smaller_area) if smaller_area > 0 else 0
    
    return containment_ratio


def filter_boxes_by_area(groups, containment_ratio_threshold):
    filtered_groups = []
    
    for group in groups:
        filtered_group = []
        
        while group:
            box, conf = group.pop(0)
            filtered_group.append((box, conf))
            
            i = 0
            while i < len(group):
                other_box, other_conf = group[i]
                containment_ratio = compute_containment_ratio(box, other_box)
                
                if containment_ratio > containment_ratio_threshold:
                    # Compare areas of box and other_box
                    box_area = compute_area(box)
                    other_box_area = compute_area(other_box)
                    
                    if box_area < other_box_area:
                        # Remove the smaller box (box) and break the loop
                        filtered_group.remove((box, conf))
                        break
                    else:
                        # Remove the smaller box (other_box)
                        group.pop(i)
                else:
                    i += 1
        
        filtered_groups.append(filtered_group)
    
    return filtered_groups


# def display_image_with_boxes(img, groups):
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(12,12))
    
#     for group in groups:
#         for box, conf in group:
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(img_rgb, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
#     plt.imshow(img_rgb)
#     plt.axis('off')
#     plt.show()

def segment_words(image_path, threshold=20, containment_ratio_threshold=0.5):
    # Run inference
    results = model(image_path)

    # Extract bounding boxes and confidence scores
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates in [x1, y1, x2, y2] format
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Combine boxes and confidences into a list of tuples
    boxes_with_conf = [(tuple(box), float(conf)) for box, conf in zip(boxes, confidences)]

    # Sort boxes by the y-center coordinate
    boxes_with_conf.sort(key=lambda x: (x[0][1] + x[0][3]) / 2)  # Sorting by the y-center coordinate

    # Group boxes based on the y-center coordinate
    groups = []
    current_group = []

    for box, conf in boxes_with_conf:
        if not current_group:
            current_group.append((box, conf))
        else:
            prev_box, _ = current_group[-1]
            prev_y_center = (prev_box[1] + prev_box[3]) / 2
            curr_y_center = (box[1] + box[3]) / 2
            if abs(curr_y_center - prev_y_center) <= threshold:
                current_group.append((box, conf))
            else:
                groups.append(current_group)
                current_group = [(box, conf)]
                
    # Add the last group if it exists
    if current_group:
        groups.append(current_group)

    # Sort each group by x1 coordinate
    for i, group in enumerate(groups):
        groups[i] = sorted(group, key=lambda x: x[0][0])  # Sorting each group by x1 coordinate

    # Apply the filtering function
    filtered_groups = filter_boxes_by_area(groups, containment_ratio_threshold)

    # Load and display the image with bounding boxes
    img = cv2.imread(image_path)
    # display_image_with_boxes(img, filtered_groups)

    return filtered_groups




def distortion_free_resize(image,img_size):
  w,h = img_size
  image = tf.image.resize(image,size=(h,w),preserve_aspect_ratio = True)

  #check the amount of padding needed to be done.
  pad_height = h - tf.shape(image)[0]
  pad_width = w - tf.shape(image)[1]

  # only necessary if you want to do same amount of padding on both sides.
  if pad_height % 2 != 0:
    height = pad_height // 2
    pad_height_top = height + 1
    pad_height_bottom = height
  else:
    pad_height_top = pad_height_bottom = pad_height//2
  if pad_width %2 != 0:
    width = pad_width//2
    pad_width_left = width + 1
    pad_width_right = width
  else:
    pad_width_left = pad_width_right = pad_width//2

  image = tf.pad(
      image,
      paddings = [
          [pad_height_top, pad_height_bottom],
          [pad_width_left, pad_width_right],
          [0,0],
      ],
  )

  image = tf.transpose(image, perm= [1,0,2])
  image = tf.image.flip_left_right(image)
  return image


def preprocess_image(image_path, img_size=(image_width,image_height)):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_png(image,1)
  image = distortion_free_resize(image,img_size)
  image = tf.cast(image,tf.float32)/255.0
  return image
  

def decode_single_prediction(pred):
  input_len = np.ones(1) * pred.shape[1]
  results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][0][:max_len]
  res = tf.gather(results, tf.where(tf.math.not_equal(results, -1)))
  return tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")


def inference_on_single_image(image_path):
    inference_image = preprocess_image(image_path)
    expanded_image = tf.expand_dims(inference_image,axis=0)
    prediction = prediction_model.predict(expanded_image)
    predicted_text = decode_single_prediction(prediction)
    return predicted_text


def save_cropped_images(image_path, filtered_groups):
    # Read the input image
    img = cv2.imread(image_path)

    # Create the main temporary directory
    temp_dir = 'temporary_images'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Initialize a list to store image paths
    image_paths = []

    # Loop through the groups and save images
    for group_index, group in enumerate(filtered_groups):
        # List to hold paths for the current line group
        line_image_paths = []
        
        for box_index, (box, _) in enumerate(group):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Crop the image
            cropped_img = img[y1:y2, x1:x2]
            
            # Convert to RGB for saving
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            
            # Define the filename for the cropped image
            filename = f'line{group_index + 1}word{box_index + 1}.png'
            file_path = os.path.join(temp_dir, filename)
            
            # Save the cropped image
            cv2.imwrite(file_path, cv2.cvtColor(cropped_img_rgb, cv2.COLOR_RGB2BGR))

            # Append the path to the current line list
            line_image_paths.append(file_path)
        
        # Append the current line paths to the main list
        image_paths.append(line_image_paths)

    # Return the list of image paths
    return image_paths



def perform_inference_on_cropped_images(cropped_image_paths, temp_dir="temporary_images"):
    final_prediction = ""  # Variable to store all the predictions

    # Loop through each line group of images
    for group_index, line_paths in enumerate(cropped_image_paths):
        print(f"Performing inference on Group {group_index + 1}...")

        # Loop through each word (cropped image) in the group
        for word_index, img_path in enumerate(line_paths):
            # Perform inference on the cropped image
            predicted_text = inference_on_single_image(img_path)
            final_prediction += predicted_text + " "
        final_prediction += "\n"

    # Delete the temporary directory and its contents after inference
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    return final_prediction  # Return the entire prediction text


def perform_prediction(image_path,threshold=20):
    filtered_groups = segment_words(image_path,threshold)
    cropped_image_paths = save_cropped_images(image_path, filtered_groups)
    digitized_text = perform_inference_on_cropped_images(cropped_image_paths)
    return digitized_text



image_path = 'inference_images/discord_test.jpeg'
# display_original_image(image_path)
digitized_text = perform_prediction(image_path)
print(digitized_text)
