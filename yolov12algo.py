⛳️ YOLOv12 Algorithm

==>Algorithm Overview:

YOLO (You Only Look Once) is a single-stage object detection algorithm that predicts bounding boxes and class probabilities directly from images in one forward pass — making it extremely fast for real-time use.

Algorithm Steps:

1) Input Image Processing:

The image is divided into an S × S grid (e.g., 640×640 resized).



2)Feature Extraction:

The backbone network ((Cross Stage Partial(CSP)-Darknet or GELAN(Generalized Efficient Layer Aggregation Network) in YOLOv12) extracts hierarchical visual features.
CSP->reduce computational cost and improve gradient flow.
GELAN->better accuracy-speed.

->edges, colors, and textures (in early layers),
->shapes, contours, and object parts (in deeper layers).
->“eyes” of your system 



3)Detection Head:

Decides what’s in the image and where.
It takes the feature maps from the backbone and predicts:  --> the object class (e.g., elephant, boar, monkey),
										   --> the bounding box coordinates, and
										   --> the confidence score.
										   
Uses FPN(Feature Pyramid Network) --> Helps the model detect objects at multiple sizes.
							Example: monkeys (small) vs elephants (large).
							
Uses PAN(Path Aggregation Network) --> Helps in bounding boxes are more accurate.



4)Bounding Box Prediction:
Each grid cell predicts: Coordinates (x, y, w, h)
Objectness score
Class probabilities (e.g., elephant, boar, monkey)



5)Non-Maximum Suppression (NMS):

Removes overlapping boxes, keeping only the most confident predictions.



6)Output:

Gives final detections with class name, bounding box, and confidence score.

======================================================================
for image in dataset:
    image = preprocess(image)
    features = Backbone(image)
    fused_features = PANet(features)
    detections = DetectionHead(fused_features)
    boxes = NonMaxSuppression(detections)
    display_results(boxes)
======================================================================


# --- YOLOv12 Animal Detection ---
from ultralytics import YOLO
import cv2

# Load trained YOLOv12 model
model = YOLO("/home/user/runs/detect/train/weights/best.pt")  # your trained model

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv12 inference
    results = model(frame)

    # Plot boxes and labels
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("YOLOv12 - Wildlife Detection", annotated_frame)

    # Break with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
