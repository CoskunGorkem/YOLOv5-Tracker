# YOLOv5-Tracker


This repository contains extra methods on top of the [ultralytics yolov5 repository](https://github.com/ultralytics/yolov5)
Extra features are:
- Light, strong custom object tracker
- Key point finder and feature matcher with optical flow
- Variance of the image laplacian to calculate image blur(To avoid possible false detections)

## Statistical Accuracy

### Object Detection Accuracy
<img src="https://user-images.githubusercontent.com/26833433/103594689-455e0e00-4eae-11eb-9cdf-7d753e2ceeeb.png" width="1000">** 
### Tracker Algorithm Accuracy
```bash
Whole pipeline is runned on 100 randomly selected videos from Berlin. 
Within each video, a json file that indicates the individual car locations is also available. (private data)
Model output is compared with ground truth data.
Tracker algorithm success calculated as 89.5% in the test. 
```

<img src="readmefiles/example.gif" width = "800" >
Due to some privacy reason, the geo location of the videos are not to be shared. But an example output of map module(not included in the repository) is represented here:
<img src="readmefiles/example_map.png" width = "700" >


## Requirements

To install the pipeline run:
```bash
$ pip install -r requirements.txt
```

## Inference
To run the inference on test video simply run:
```bash
$ python analysis/main.py
```
To run the inference on a defined video simply run:
```bash
$ python analysis/main.py --source /path/to/video.mp4
```


