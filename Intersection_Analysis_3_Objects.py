# Please read the following before using this script:
### In this script, the names of the objects that the YOLO model has been trained to detect are defined as object_1, object_2, and object_3. These are set to ‘Mouse’, ‘R_PetriDish’, and ‘L_PetriDish’, respectively.
### If you want to change these names, you can replace ‘Mouse’, ‘R_PetriDish’, and ‘L_PetriDish’ with the new names of the classes that your model has been trained to detect.
### Please note that the names used in the script must match the class names that your model has been trained on. The class names are typically defined in the data configuration file used for training the YOLO model. If the names do not match, the script will not be able to correctly identify and process the detected objects.
### Define the names of your objects in lines 58-61.

# Begining of the code: 
from ultralytics import YOLO
import cv2
import numpy as np
import csv

# Function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xi1 = max(x1, box2[0])
    yi1 = max(y1, box2[1])
    xi2 = min(x2, box2[2])
    yi2 = min(y2, box2[3])
    inter_area = max(xi2-xi1, 0) * max(yi2-yi1, 0)
    box1_area = (x2-x1)*(y2-y1)
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

# Load Model
model = YOLO(r"D:\YOLOv8_Object_Mouse_Interaction\runs\detect\train\weights\best.pt")

# Load video
video_path=r"D:\YOLOv8_Object_Mouse_Interaction\2024-06-01 21-40-33.mp4"
cap= cv2.VideoCapture(video_path)
ret = True

# Initialize interaction counter and duration for both objects
interaction_counter_2 = 0
interaction_duration_2 = 0
interaction_counter_3 = 0
interaction_duration_3 = 0

# Initialize flag for ongoing interaction for both objects
ongoing_interaction_2 = False
ongoing_interaction_3 = False

# Initialize frame counter
frame_counter = 0

# Set the minimum interaction duration threshold
min_interaction_duration = 5  # Change this to the desired value

# Initialize list to store interaction data for both objects
interaction_data_2 = []
interaction_data_3 = []

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',  fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Define the objects to track
object_1 = 'Mouse'  # Change this to the name of the first object
object_2 = 'R_PetriDish'  # Change this to the name of the second object
object_3 = 'L_PetriDish'  # Change this to the name of the third object

# Read frames
while ret:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Increment frame counter
    frame_counter += 1

    # run model predictions on the incoming frames
    results = model(frame, max_det=3)

    # Check if there are any detections in the current frame
    if len(results) > 0:
        object_1_bbox = None
        object_2_bbox = None
        object_3_bbox = None
        for result in results:
            # Extract bounding boxes, classes, names, and confidences
            boxes = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist()
            names = result.names
            confidences = result.boxes.conf.tolist()

            # Iterate through the results
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                confidence = conf
                detected_class = cls
                name = names[int(cls)]

                # Check if the object is Object1, Object2 or Object3 and get the corresponding bounding box
                if name == object_1:
                    object_1_bbox = [x1, y1, x2, y2]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, object_1, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                elif name == object_2:
                    object_2_bbox = [x1, y1, x2, y2]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, object_2, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                elif name == object_3:
                    object_3_bbox = [x1, y1, x2, y2]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, object_3, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        # If Object1 and either Object2 or Object3 were detected in the frame, check if they are interacting
        if object_1_bbox is not None:
            if object_2_bbox is not None:
                iou_2 = calculate_iou(object_1_bbox, object_2_bbox)
                interacting_2 = iou_2 > 0.01
            if object_3_bbox is not None:
                iou_3 = calculate_iou(object_1_bbox, object_3_bbox)
                interacting_3 = iou_3 > 0.01

            if interacting_2 and not ongoing_interaction_2:
                # Start of a new interaction with Object2
                ongoing_interaction_2 = True
                interaction_counter_2 += 1
                interaction_duration_2 = 1
            elif interacting_2 and ongoing_interaction_2:
                # Continuation of an ongoing interaction with Object2
                interaction_duration_2 += 1
            elif not interacting_2 and ongoing_interaction_2:
                # End of an ongoing interaction with Object2
                ongoing_interaction_2 = False
                if interaction_duration_2 >= min_interaction_duration:
                    print(f'Interaction with {object_2} {interaction_counter_2} lasted for {interaction_duration_2} frames')
                    interaction_data_2.append([interaction_counter_2, interaction_duration_2])
                interaction_duration_2 = 0

            if interacting_3 and not ongoing_interaction_3:
                # Start of a new interaction with Object3
                ongoing_interaction_3 = True
                interaction_counter_3 += 1
                interaction_duration_3 = 1
            elif interacting_3 and ongoing_interaction_3:
                # Continuation of an ongoing interaction with Object3
                interaction_duration_3 += 1
            elif not interacting_3 and ongoing_interaction_3:
                # End of an ongoing interaction with Object3
                ongoing_interaction_3 = False
                if interaction_duration_3 >= min_interaction_duration:
                    print(f'Interaction with {object_3} {interaction_counter_3} lasted for {interaction_duration_3} frames')
                    interaction_data_3.append([interaction_counter_3, interaction_duration_3])
                interaction_duration_3 = 0

            # Display the interaction status and frame number on the video frame
            if interacting_2 or interacting_3:
                cv2.putText(frame, 'Interactions: ' + object_2 + ' ' + str(interaction_counter_2) + ', ' + object_3 + ' ' + str(interaction_counter_3), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Frame: ' + str(frame_counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Write the frame to the output video file
                out.write(frame)

                # Display the frame
                #cv2.imshow('Video', frame)

    # Write the frame to the output video file
    out.write(frame)
    # Break the loop if 'q' is pressed
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

cap.release()
out.release()
cv2.destroyAllWindows()

# Print the total number of interactions
print('Total number of interactions with ' + object_2 + ':', interaction_counter_2)
print('Total number of interactions with ' + object_3 + ':', interaction_counter_3)

# Write the interaction data to CSV files
with open('interaction_data_' + object_2 + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Interaction", "Duration"])
    writer.writerows(interaction_data_2)

with open('interaction_data_' + object_3 + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Interaction", "Duration"])
    writer.writerows(interaction_data_3)
