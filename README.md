[![DOI](https://zenodo.org/badge/823833552.svg)](https://zenodo.org/doi/10.5281/zenodo.12684627)

# YouTube Video Tutorial for this Project: (Click the Image Below)
[![Youtube Tutorial](https://github.com/farhanaugustine/NovelObject_Interaction_YOLOv8/assets/54376988/a22aa44a-87d3-49be-b028-abe0221ba813)](https://youtu.be/0iptIs9ccGg?si=S0OfYyacMZSSL82n)


# Novel Object_Mouse_Interaction_ w/ YOLOv8

### Mouse-Object Interaction Detection

### These scripts use a YOLO model to detect interactions between a mouse and object(s) in a video. An interaction is defined as the mouse and the object having an Intersection over Union (IoU) of their bounding boxes greater than a specified threshold.

 $\textcolor{yellow}{There\ are\ two\ python\ scripts\ in\ this\ project\ :}$
1. Script titled $\textcolor{yellow}{NovelObjectDetection.py}$ is meant for use with two YOLOv8 object classes (Mouse and NovelObject).
2. Script titled $\textcolor{yellow}{Intersection\ _\ Analysis\ _\ 3\ _\ Objects.py\}$ is meant for use with three YOLOv8 object classes (Object 1, Object 2, and Object 3). Users can change the names of these classes (please see instructions on the script itself).

# $\textcolor{yellow}{Special\ Note:}$
$\textcolor{yellow}{The\ script\ works\ by\ running\ the\ YOLO\ model\ on\ each\ frame\ of\ the\ video\ and\ checking\ if\ both\ a\ Mouse\ and\ a\ NovelObject\ are\ detected\ in\ the\ frame\ .}$
$\textcolor{yellow}{If\ both\ objects\ are\ detected\ ,\ it\ calculates\ the\ Intersection\ over\ Union\ (IoU)\ of\ their\ bounding\ boxes\ to\ determine\ if\ they\ are\ interacting\ .\}$
$\textcolor{yellow}{If\ an\ interaction\ is\ detected\ ,\ the\ script\ increments\ an\ interaction\ counter\ and\ duration\ ,\ and\ writes\ the\ frame\ to\ an\ output\ video\ file\ .\}$

$\textcolor{yellow}{However\ ,\ if\ no\ interaction\ is\ detected\ in\ a\ frame\ (i.e.\ ,\ the\ IoU\ of\ the\ bounding\ boxes\ is\ less\ than\ a\ threshold\ ,\ or\ only\ one\ or\ none\ of\ the\ objects\ are\ detected\ )\,}$ 
$\textcolor{yellow}{the\ script\ does\ not\ write\ the\ frame\ to\ the\ output\ video\ file\ .\ Therefore\ ,\ the\ output\ video\ will\ only\ contain\ frames\ where\ an\ interaction\ between\ the\ ‘Mouse’\ and\ the\ ‘NovelObject’\ is\ observed\ .\}$

If you want to save all frames, including those without interaction, you would need to move the line ***``out.write(frame)``*** outside of the ***``if interacting:``*** block of code so that it is executed for every frame, regardless of whether an interaction is detected or not. Please remember to adjust your code accordingly if you decide to make this change.


# Requirements

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

You can customize the script by changing the following parameters for each script:

    video_path: The path to the video file.
    model_path: The path to the YOLO model weights file.
    iou_threshold: The IoU threshold for considering the mouse and the object to be interacting. Default is 0.01.
    min_interaction_duration: The minimum number of frames for an interaction to be counted. Default is 5.

Note

The output video only contains the frames where an interaction is detected. This is because the script is designed to focus on the interactions. If you want the output video to contain all frames, you can modify the script by moving the `out.write(frame)` outside of the loop, as mentioned above.
