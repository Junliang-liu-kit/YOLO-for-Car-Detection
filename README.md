YOLO for Car Detection
===============
object detection using the very powerful YOLO model. 'YOLO' ("you only look once") is a popular algoritm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes. 

## Background
The background is about a car detection system of the self-driving car. To collect data, you've mounted a camera to the hood (meaning the front) of the car, which takes pictures of the road ahead every few seconds while you drive around.
![video](/git_images/road_video_compressed2.mp4)
<br>Pictures taken from a car-mounted camera while driving around Silicon Valley, which is given from [drive.ai](https://www.drive.ai/)

## Bounding Boxes in object dectection

The detected car is marked as a box. We have gathered all these images into a folder and have labelled them by drawing bounding boxes around every car we found. Here's an example of what the bounding boxes look like:
![image](/git_images/box_label.png)

## Model details
Because of very computationally expensive to train, here we will load pre-trained weights of the YOLO model to use. About this pre trained YOLO model:
* The input is a batch of images of shape ('m, 608, 608, 3')
* The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers  (𝑝𝑐,𝑏𝑥,𝑏𝑦,𝑏ℎ,𝑏𝑤,𝑐)  as explained above. We expand  𝑐  into an 80-dimensional vector, so each bounding box is then represented by '85 numbers'. The 80 different classes listed in [coco_classes.txt](/model_data/coco_classes.txt)
![image](/git_images/architecture.png)
<br>We will use 5 anchor boxes. Sothe YOLO architecture as the following: 
<br>'IMAGE' (m, 608, 608, 3) -> 'DEEP CNN' -> 'ENCODING' (m, 19, 19, 5, 85).


<br>'Anchor boxes' is a method in object-detection solving the problem, when the middle points of two or more objects set in the same grid cell. Here we are using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.
![image](/git_images/flatten.png)
<br>For simplicity, we will flatten the last two last dimensions of the shape (19, 19, 5, 85) encoding. So the output of the Deep CNN is (19, 19, 425).


<br>For each box (of each cell) we will compute the following elementwise product and extract a probability that the box contains a certain class('Scores').
![image](/git_images/probability_extraction.png)


<br>Here's one way to visualize what YOLO is predicting on an image:
![image](/git_images/anchor_map.png)
<br>Each cell gives you 5 boxes. In total, the model predicts: 19x19x5 = 1805 boxes just by looking once at the image


## Filtering with a threshold
In the figure above, we plotted only boxes that the model had assigned a high probability to, but this is still too many boxes. We would like to filter the algorithm's output down to a much smaller number of detected objects using non-max suppression. Specifically, we'll carry out these steps:

* Get rid of boxes with a low threshold(meaning, the box is not very confident about detecting a class)
* Select only one box when several boxes overlap with each other and detect the same object('Non-max suppression')

## Non-max suppression
Even after filtering by thresholding over the classes scores, it often still end up a lot of overlapping boxes. A second filter for selecting the right boxes is called 'non-maximum suppression' (NMS).
![image](/git_images/non-max-suppression.png)

<br>Non-max suppression uses the very important function called "'Intersection over Union'", or IoU.
![image](/git_images/iou.png)

## Conclusion

the detection output of one image
![image](/out/test.jpg)

<br>If we were to run the session in a for loop over all images, we can get:
![image](/git_images/pred_video_compressed2.mp4)
<br>Predictions of the YOLO model on pictures taken from a camera while driving around the Silicon Valley, which is given from [drive.ai](https://www.drive.ai/)

## References 
The ideas presented in this notebook came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's github repository. The pretrained weights used in this exercise came from the official YOLO website.

<br>Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection (2015)](https://arxiv.org/abs/1506.02640)
<br>Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger (2016)](https://arxiv.org/abs/1612.08242)
<br>Allan Zelener - YAD2K: [Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
<br>The official YOLO website [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
