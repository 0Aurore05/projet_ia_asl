from ultralytics import YOLO
import numpy
import cv2

model = YOLO('best.pt')

### IOU
# Area of overlap / Area of union
def iou(xA, yA, xB, yB):

  boxA = [xA, yA, xB, yB]

  if res[0].boxes:
    bb = res[0].boxes[0].xyxy.numpy()[0]
    pxA = int(float(bb[0]))
    pyA = int(float(bb[1]))
    pxB = int(float(bb[2]))
    pyB = int(float(bb[3]))

    boxB = [pxA, pyA, pxB, pyB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    
    # draw IoU on image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(
      image,
      "IoU: {:.4f}".format(iou),
      (20, size_y - 20),
      font,
      0.8,
      (255, 255, 255),
      1,
    )

for i in range(1, 261):

  image_name = "annoted/test/images/frame_"+ str(i) + ".jpg"
  label_name = "annoted/test/labels/frame_"+ str(i) + ".txt"

  ### YOLO drawing
  res = model(image_name, save=False)
  image = res[0].plot()


  ### GROUND TRUTH DRAWING
  # get image size to adjust bounding box float values to pixel values
  size_x = image.shape[1]
  size_y = image.shape[0]

  # open label file with bounding box values
  # converting (x_center, y_center, width, height) to a (xA, yA, xB, yB) format
  f = open(label_name)
  bb = f.read().split()
  if bb:
    label = bb[0]
    x = float(bb[1]) * size_x
    y = float(bb[2]) * size_y
    width = float(bb[3]) * size_x
    height = float(bb[4]) * size_y

    xA = int(x - width / 2)
    yA = int(y - height / 2)
    xB = int(x + width / 2)
    yB = int(y + height / 2)

    start_point = (xA, yA)
    end_point = (xB, yB)

    # draw box
    cv2.rectangle(image, start_point, end_point, (50, 50, 50), 3)

    # draw class name + confidence
    class_list = ["L", "B", "C", "7"]

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(
      image,
      class_list[int(label)-1],
      (xA + int(width) - 25, yA + 23),
      font,
      0.7,
      (50, 50, 50),
      1,
    )

    iou(xA, yA, xB, yB)

  cv2.imwrite("RESULTS_IOU/frame_" + str(i) + ".jpg", image)



