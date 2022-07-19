
**Author: Adedamola Sode
**Email: adedamola.sode@gmail.com

# Working With Tracking Algorithms 
This project is used to test and implement simple trackers after developing object detection solutions to solve particular problems. The test file is a video of moving traffic with the main objective being to track the vehicles

## Requirements
1.OpenCV
2.Numpy
3.Pytorch

## Tracking using the difference in euclidean distances
Using the outsourced <a href=https://github.com/AsimovNo9/ObjectTracking/blob/main/modules/tracker.py>tracker.py</a> module. 

1. To run:

```bash
python main.py -m normal
```

## Tracking using OpenCV functions
Implementing the tracking functions such as KCF in Open CV and deploying them for tracking purposes.
1. To run:

```bash
python main.py -m kcf
```