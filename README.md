# NovelObject_Interaction_YOLOv8

Mouse-Object Interaction Detection

This script uses a YOLO model to detect interactions between a mouse and an object in a video. An interaction is defined as the mouse and the object having an Intersection over Union (IoU) of their bounding boxes greater than a specified threshold.
Requirements

    Python 3.6 or later
    OpenCV
    Ultralytics YOLOv3

Usage

    Set up the environment: Make sure you have Python 3.6 or later installed. You also need to install the OpenCV and Ultralytics YOLOv3 libraries. You can install them with pip:

    pip install opencv-python-headless
    pip install yolov3

    Prepare the model: The script uses a YOLO model for object detection. You need to train this model on your data so that it can detect the mouse and the object. Once the model is trained, save the weights to a .pt file.

    Prepare the video: The script analyzes a video file. The video should be clear enough for the model to detect the mouse and the object.

    Run the script: You can run the script with Python:

    python script.py

    Check the results: The script will display the video with bounding boxes around the detected mouse and object. It will also print the number of interactions and the duration of each interaction in frames. The output video will only contain the frames where an interaction is detected.

Customization

You can customize the script by changing the following parameters:

    video_path: The path to the video file.
    model_path: The path to the YOLO model weights file.
    iou_threshold: The IoU threshold for considering the mouse and the object to be interacting. Default is 0.01.
    min_interaction_duration: The minimum number of frames for an interaction to be counted. Default is 5.

Note

The output video only contains the frames where an interaction is detected. This is because the script is designed to focus on the interactions. If you want the output video to contain all frames, you can modify the script by moving the `out.write(frame)` outside of the loop.
