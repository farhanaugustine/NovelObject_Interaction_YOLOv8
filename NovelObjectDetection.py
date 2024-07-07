### The script is designed to detect interactions between a ‘Mouse’ and a ‘NovelObject’ in a video.
### Change the names in * lines 82-90 * with the names of your own objects. (NOTE: Names used in the script must match the class names on which your model has been trained. The class names are typically defined in the data configuration file used for training the YOLO model. If the names do not match, the script will not be able to identify and process the detected objects correctly)
### Also, remember to change the labels in the cv2.putText function calls accordingly to display the correct labels on the video frames.

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
model = YOLO(r"C:\Users\Lin\Desktop\Test_NovelObjectRecognition\runs\detect\train\weights\best.pt")

# Load video
video_path=r"C:\Users\Lin\Desktop\Test_NovelObjectRecognition\demo6_video.mp4"
cap= cv2.VideoCapture(video_path)
ret = True

# Initialize interaction counter and duration
interaction_counter = 0
interaction_duration = 0

# Initialize flag for ongoing interaction
ongoing_interaction = False

# Initialize frame counter
frame_counter = 0

# Set the minimum interaction duration threshold
min_interaction_duration = 5  # Change this to the desired value

# Initialize list to store interaction data
interaction_data = []

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',  fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Read frames
while ret:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Increment frame counter
    frame_counter += 1

    # run model predictions on the incoming frames
    results = model(frame, max_det=2)

    # Check if there are any detections in the current frame
    if len(results) > 0:
        mouse_bbox = None
        novel_object_bbox = None
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

                # Check if the object is a Mouse or a NovelObject and get the corresponding bounding box
                if name == 'Mouse':
                    mouse_bbox = [x1, y1, x2, y2]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, 'Mouse', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                elif name == 'NovelObject':
                    novel_object_bbox = [x1, y1, x2, y2]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, 'NovelObject', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # If both a Mouse and a NovelObject were detected in the frame, check if they are interacting
        if mouse_bbox is not None and novel_object_bbox is not None:
            iou = calculate_iou(mouse_bbox, novel_object_bbox)
            interacting = iou > 0.01

            if interacting and not ongoing_interaction:
                # Start of a new interaction
                ongoing_interaction = True
                interaction_counter += 1
                interaction_duration = 1
            
            elif interacting and ongoing_interaction:
                # Continuation of an ongoing interaction
                interaction_duration += 1
            elif not interacting and ongoing_interaction:
                # End of an ongoing interaction
                ongoing_interaction = False
                if interaction_duration >= min_interaction_duration:
                    print(f'Interaction {interaction_counter} lasted for {interaction_duration} frames')
                    interaction_data.append([interaction_counter, interaction_duration])
                interaction_duration = 0

            # Display the interaction status and frame number on the video frame
            if interacting:
                cv2.putText(frame, 'Interactions: ' + str(interaction_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Frame: ' + str(frame_counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Write the frame to the output video file
                out.write(frame)

                # Display the frame
                #cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

cap.release()
out.release()
cv2.destroyAllWindows()

# Print the total number of interactions
print('Total number of interactions:', interaction_counter)

# Write the interaction data to a CSV file
with open('interaction_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Interaction", "Duration"])
    writer.writerows(interaction_data)
