from ultralytics import YOLO

model = YOLO('yolov8n.pt') # load pretrained YOLO model
# model = YOLO('yolov8n.yaml') # create new YOLO model
# results = model.train(data='coco128.yaml', epochs=3) # train model with default YOLO dataset

# ---

# results = model.train(data='our_training_data.yaml', epochs=3) # train model with our lab equipment dataset

results = model('https://ultralytics.com/images/bus.jpg') # detect objects in image, output info to console

# ---

# results = model.val() # evaluate performance on validation set
# success = model.export(format='onnx') # export to ONNX format
