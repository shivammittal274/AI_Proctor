import scipy
from keras.models import load_model
from PIL import Image
from mtcnn.mtcnn import MTCNN
import os
from os import listdir
from os.path import isdir
from numpy import asarray
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from random import choice
# from matplotlib import pyplot


model = load_model("./models/facenet_keras.h5")

# summarize input and output shape
print(model.inputs)
print(model.outputs)

def extract_face(filename, required_size=(160, 160)):

	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)

	detector = MTCNN()
	results = detector.detect_faces(pixels)
	if(len(results)>1):
         print('Multiple faces Present')
	elif(len(results)==0):
         print('No face Present')
	x1, y1, width, height = results[0]['box']

	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

	face = pixels[y1:y2, x1:x2]

	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
  
	return face_array


def get_embedding(model, face_pixels):
      
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
  # make prediction to get embedding
	yhat = model.predict(samples)
  
	return yhat[0]


# Place image path at t=0
face_array = extract_face("../input/multifaces/63063.jpg")
embedding = get_embedding(model, face_array)
print(embedding.shape)

face_array2=extract_face("../input/datasets-for-face-recog-task/Dataset_GOI-20210429T174054Z-001/Dataset_GOI/train/Arya/02.jpg")
embedding2 = get_embedding(model, face_array2)
print(embedding2.shape)

sim = 1 - scipy.spatial.distance.cosine(embedding, embedding2)

print(sim)

if(sim<0.75):
    print("Wrong Face")
else:
    print("Good going")
