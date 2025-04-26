from flask import Flask, request, jsonify, render_template, send_file
import uuid
import os
import cv2
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
from ultralytics import YOLO
import shutil
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Initialize models and character mapping
print("Loading models...")
yolo_model = YOLO('yolo_best60epoch.pt')

with open('characters_list.pkl', 'rb') as file:
    ff_loaded = pickle.load(file)

char_to_num = StringLookup(vocabulary=ff_loaded, mask_token=None)
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


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
  

# Load CRNN model
custom_objects = {'CTCLayer': CTCLayer}
with keras.utils.custom_object_scope(custom_objects):
    crnn_model = keras.models.load_model('handwriting_recognizer50.h5')

prediction_model = keras.models.Model(
    crnn_model.get_layer(name="image").input,
    crnn_model.get_layer(name="dense2").output
)
print("All models loaded successfully!")

# OCR Processing Functions
def compute_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def compute_containment_ratio(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    smaller_area = min(compute_area(boxA), compute_area(boxB))
    return interArea / float(smaller_area) if smaller_area > 0 else 0

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
                    box_area = compute_area(box)
                    other_box_area = compute_area(other_box)
                    if box_area < other_box_area:
                        filtered_group.remove((box, conf))
                        break
                    else:
                        group.pop(i)
                else:
                    i += 1
        filtered_groups.append(filtered_group)
    return filtered_groups

def segment_words(image_path, threshold=20, containment_ratio_threshold=0.5):
    results = yolo_model(image_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    boxes_with_conf = [(tuple(box), float(conf)) for box, conf in zip(boxes, confidences)]
    boxes_with_conf.sort(key=lambda x: (x[0][1] + x[0][3]) / 2)
    
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
    if current_group:
        groups.append(current_group)
    
    for i, group in enumerate(groups):
        groups[i] = sorted(group, key=lambda x: x[0][0])
    
    filtered_groups = filter_boxes_by_area(groups, containment_ratio_threshold)
    return filtered_groups

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]
    
    if pad_height % 2 != 0:
        pad_height_top = pad_height // 2 + 1
        pad_height_bottom = pad_height // 2
    else:
        pad_height_top = pad_height_bottom = pad_height // 2
        
    if pad_width % 2 != 0:
        pad_width_left = pad_width // 2 + 1
        pad_width_right = pad_width // 2
    else:
        pad_width_left = pad_width_right = pad_width // 2
        
    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(128, 32)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def decode_single_prediction(pred):
    input_len = np.ones(1) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][0][:27]
    res = tf.gather(results, tf.where(tf.math.not_equal(results, -1)))
    return tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")

def inference_on_single_image(image_path):
    inference_image = preprocess_image(image_path)
    expanded_image = tf.expand_dims(inference_image, axis=0)
    prediction = prediction_model.predict(expanded_image)
    return decode_single_prediction(prediction)

def save_cropped_images(image_path, filtered_groups, temp_dir):
    img = cv2.imread(image_path)
    image_paths = []
    
    for group_index, group in enumerate(filtered_groups):
        line_image_paths = []
        for box_index, (box, _) in enumerate(group):
            x1, y1, x2, y2 = map(int, box)
            cropped_img = img[y1:y2, x1:x2]
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            filename = f'line{group_index+1}word{box_index+1}.png'
            file_path = os.path.join(temp_dir, filename)
            cv2.imwrite(file_path, cv2.cvtColor(cropped_img_rgb, cv2.COLOR_RGB2BGR))
            line_image_paths.append(file_path)
        image_paths.append(line_image_paths)
    return image_paths

def perform_prediction(image_path, temp_dir):
    filtered_groups = segment_words(image_path)
    cropped_image_paths = save_cropped_images(image_path, filtered_groups, temp_dir)
    
    final_prediction = ""
    for line_paths in cropped_image_paths:
        for img_path in line_paths:
            predicted_text = inference_on_single_image(img_path)
            final_prediction += predicted_text + " "
        final_prediction += "\n"
    
    return final_prediction.strip()

def deskew_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel)
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    angles = [cv2.minAreaRect(c)[-1] for c in contours if cv2.minAreaRect(c)[-1] not in [90.0, -0.0]]
    mid_angle = sorted(angles)[len(angles) // 2] if angles else 0
    if mid_angle > 45:
        mid_angle = -(90 - mid_angle)

    height, width = img.shape[:2]
    m = cv2.getRotationMatrix2D((width / 2, height / 2), mid_angle, 1)
    deskewed = cv2.warpAffine(img, m, (width, height), borderValue=(255, 255, 255))

    cv2.imwrite(output_path, deskewed)
    return output_path


@app.route('/api/deskew', methods=['POST'])
def deskew_image_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_dir = os.path.join('temp', str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        input_path = os.path.join(temp_dir, 'input.png')
        output_path = os.path.join(temp_dir, 'deskewed.png')
        file.save(input_path)

        deskewed_path = deskew_image(input_path, output_path)
        
        # Read the file into memory before cleaning up
        with open(deskewed_path, 'rb') as f:
            image_data = f.read()
            
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Return the file from memory instead of from disk
        return send_file(
            io.BytesIO(image_data),
            mimetype='image/png',
            as_attachment=False
        )

    except Exception as e:
        # Make sure we clean up on errors too
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

# Add this route to your Flask application to return bounding boxes
@app.route('/api/ocr', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    temp_dir = os.path.join('temp', str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        input_path = os.path.join(temp_dir, 'input.png')
        file.save(input_path)
        
        # Segment words using YOLO
        filtered_groups = segment_words(input_path)
        
        # Process for OCR
        result = perform_prediction(input_path, temp_dir)
        
        # Check if we should return bounding boxes for visualization
        return_boxes = request.headers.get('X-Return-Boxes', '').lower() == 'true'
        response_data = {'text': result}
        
        if return_boxes:
            # Convert bounding box objects to lists for JSON serialization
            boxes_list = []
            for group in filtered_groups:
                group_boxes = []
                for box, _ in group:
                    group_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                boxes_list.append(group_boxes)
            response_data['boxes'] = boxes_list
        
        shutil.rmtree(temp_dir)
        return jsonify(response_data)
    
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)