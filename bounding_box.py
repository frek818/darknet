from darknetpy import Detector
import cv2
import argparse


def add_boxes(img, boxes):
    for box in boxes:
        t = int(box['top'])
        l = int(box['left'])
        b = int(box['bottom'])
        r = int(box['right'])
        img = cv2.rectangle(img, (t,l), (b,r), (0,255,0), 2)
    return img

if __name__ == '__main__':
    detector = Detector(
                 '/home/herman/darknet/cfg/coco.data',
                 '/home/herman/darknet/cfg/yolov3.cfg',
                 '/home/herman/darknet/data/yolov3.weights')

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, help="Input video file path")

    args = parser.parse_args()

    in_file = args.infile;

    cap = cv2.VideoCapture(in_file)
    ok, frame = cap.read()
    while ok:
        frame = cv2.resize(frame, None, fx=0.15, fy=0.15)
        cv2.imwrite("/tmp/frame.jpg", frame)
        boxes = detector.detect("/tmp/frame.jpg")
        frame = add_boxes(frame, boxes)

        cv2.imshow("Detected", frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
