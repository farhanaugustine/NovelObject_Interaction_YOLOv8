1. Create Anaconda Environment 

`conda create -n yolov8 python=3.9`


2. Extract Frames from Behavior Videos

` ffmpeg -i "input.mp4" -vf "fps=1/2" -frames:v 40 output_%02d.png `


3. Create a New Folder (label: "images") and Move Extracted Frames into the "images" folder
       3.1. Create another folder and label it "labels" 


4. Install Image Annotation Tool

` pip install labelImg ` or ` pip install labelme `


5.  Run Annotation Tool

`labelImg ` or ` labelme `
Make sure that you set the output directory to the label folder.

6. Annotate your Object by creating a rectangle and labeling it. 


7. Install ultralytics

` pip install ultralytics `


8. Upgrade/Install Pytorch 

Copy code from here and install it into your environment: ` https://pytorch.org/get-started/locally/ `

Example: ` pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 `


9. Create "val" and "train" folders

10. Subdivide "val" folder by making two more folders (i.e., "images" and "labels" folders, respectively; these will be used for evaluation of model after training).

         10.1. Move some of your training images from the training folder into the "images" folder under the val directory. Move the corresponding labels from the training folder into the "labels" folder under the val directory. [delete the images and labels that you are moving from training directory to the val subfolders so that the algorithm won't see these images or labels while training]  


11. Create "dataset.yaml" using text editor [copy the code below into "data_custom.yaml" file using text editor]

```
train: "Path to your train images folder"
val: "Path to your val folder"
nc: 3 #change this number to match the number of objects you have in your training data.

names: ["Mouse", "object1", "object2",...]
```


12. Run YOLO for training on your images

` yolo task=detect mode=train epochs=500 data=dataset.yaml model=yolov8m.pt imgsz=640 batch=-1 fliplr=0.8 ` 
More information about yolo training hyperparameters: https://docs.ultralytics.com/guides/hyperparameter-tuning/

Run YOLO Custom Model on New Videos with Labeled Predictions

` yolo task=detect mode=predict model="Model_path.pt" show=True conf=0.5 save_txt=True source="path_to_.mp4" `

