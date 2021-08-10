import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

labels = open('../input/yolo-coco-data/coco.names').read().strip().split('\n')

print(labels)
weights_path = '../input/yolo-coco-data/yolov3.weights'
configuration_path = '../input/yolo-coco-data/yolov3.cfg'

probability_minimum = 0.3
threshold = 0.3

network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)

layers_names_all = network.getLayerNames()

layers_names_output = [layers_names_all[i[0] - 1]
                       for i in network.getUnconnectedOutLayers()]
print(layers_names_output)

image_input = cv2.imread('../input/image-2/WIN_20210804_19_25_33_Pro.jpg')

image_input_shape = image_input.shape

print(image_input_shape)


blob = cv2.dnn.blobFromImage(
    image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)


network.setInput(blob)
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()

print('YOLO v3 took {:.5f} seconds'.format(end - start))

confidences = []
class_numbers = []

h, w = image_input_shape[:2]

print(h, w)

for result in output_from_network:
    # Going through all detections from current output layer
    for detection in result:

        scores = detection[5:]
        class_current = np.argmax(scores)

        confidence_current = scores[class_current]

        if confidence_current > probability_minimum:

            box_current = detection[0:4] * np.array([w, h, w, h])

            x_center, y_center, box_width, box_height = box_current.astype(
                'int')
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            confidences.append(float(confidence_current))
            class_numbers.append(class_current)

objects = []
for i in range(len(class_numbers)):
    objects.append(labels[int(class_numbers[i])])

print('cell phone' in objects)
print('book' in objects)

confidences

