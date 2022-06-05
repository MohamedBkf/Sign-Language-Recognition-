import tensorflow.python.keras.models as tf
import numpy as np
import cv2
import pyttsx3 as ttx
parleur=ttx.init()
model=tf.load_model("")
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
np.set_printoptions(suppress=True)
capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
tableau_gestes=[]
nominalisation="abcd1.txt"
with open(nominalisation,'rt') as reading:
    nominalisation=reading.read().rstrip('\n').rsplit('\n')
ma_liste=[]
for i,mot in enumerate(nominalisation):
    mot=mot.replace(str(i),"")
    ma_liste.append(mot)
nominalisation=ma_liste
#nominalisation devient un tableau contenant toutes les données dont on aura besoin
i=0
prev=0
fps=0
import time as t
while True:
      _,img=capture.read()
      cadre= cv2.flip(img,1)
      cadre= cadre[80:360, 220:530]
      cadre= cv2.resize(cadre, (224, 224))
      cadre_tableau = np.asarray(cadre)
      normalisation_cadre_tableau = (cadre_tableau.astype(np.float32) / 127.0) - 1
      data[0] = normalisation_cadre_tableau
      prediction = model.predict(data).flatten()
      indice = np.argmax(prediction)
      fps=int(1/(t.time()-prev))
      prev=t.time()
      print(fps)
      i+=1
      if i==30:
          i=0
          if prediction[indice]>=0.9:
            cv2.putText(img,str(nominalisation[indice]),(50,70),cv2.FONT_HERSHEY_SIMPLEX,1,(150,0,0),1)
            parleur.say(str(nominalisation[indice]))
            parleur.runAndWait()
            parleur.stop()
      cv2.imshow('Frame',img )
      if cv2.waitKey(1) and 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()





# Get model details
input_details = ss.get_input_details()
output_details = ss.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname):  # This is a TF2 model
  boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
  boxes_idx, classes_idx, scores_idx = 0, 1, 2

nominalisation="abcd1.txt"
with open(nominalisation,'rt') as reading:
    nominalisation=reading.read().rstrip('\n').rsplit('\n')
ma_liste=[]
for i,mot in enumerate(nominalisation):
    mot=mot.replace(str(i),"")
    ma_liste.append(mot)
nominalisation=ma_liste



cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:

  t1,t2=cap.read()

  # Acquire frame and resize to expected shape [1xHxWx3]
  frame = t2.copy()
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame_resized = cv2.resize(frame_rgb, (width, height))
  input_data = np.expand_dims(frame_resized, axis=0)

  # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
  if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

  # Perform the actual detection by running the model with the image as input
  ss.set_tensor(input_details[0]['index'], input_data)
  ss.invoke()

  # Retrieve detection results
  boxes = ss.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates of detected objects
  classes = ss.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
  scores = ss.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects

  # Loop over all detections and draw detection box if confidence is above minimum threshold
  for i in range(len(scores)):
      if ((scores[i] > 0.7) and (scores[i] <= 1.0)):
          # Get bounding box coordinates and draw box
          # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
          ymin = int(max(1, (boxes[i][0] * 640)))
          xmin = int(max(1, (boxes[i][1] * 480)))
          ymax = int(min(640, (boxes[i][2] * 640)))
          xmax = int(min(480, (boxes[i][3] * 480)))

          cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)



  print('hy')
  cv2.imshow('Object detector', frame)


  if cv2.waitKey(1) & 0xff== ord('q'):
    break

# Clean up
cap.release()
cv2.destroyAllWindows()
