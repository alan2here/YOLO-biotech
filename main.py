from ultralytics import YOLO

# pip install ultralytics
# to do so "pip install torch" or "pip install torch==1.8.0", doesn't work, pip cannot find torch at all

# ---

# Load a pretrained YOLO model
model = YOLO('yolov8n.pt')

# Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# Train a model using a dataset
results = model.train(data='coco128.yaml', epochs=3)

# Perform object detection on an image using a model
results = model('https://ultralytics.com/images/bus.jpg')

# ---

# Evaluate a model's performance on a validation set
# results = model.val()

# Export a model to ONNX format
# success = model.export(format='onnx')
