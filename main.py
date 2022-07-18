from modules.detector import *
import sys
import getopt

opts, args = getopt.getopt(sys.argv[1:], "m:", ["mode"])

for opt, arg in opts:
    if opt == "-m":
        mode = arg

if mode == "kcf":
    Dec_ = Detector("yolov5n6.pt", "./data/traffic.mp4")
    print(mode)
    Dec_.detect_kcf()

if mode == "normal":
    Dec_ = Detector("yolov5n6.pt", "./data/traffic.mp4")
    Dec_.detect()
