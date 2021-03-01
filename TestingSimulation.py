print('Setting UP')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()
#print(sio)

app = Flask(__name__)
maxSpeed = 20


def preProcess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    #print(data)
    #print(dataA)
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])
    steering = float(model.predict(image))
    kp = 1
    ki = 0.001
    kd = 0.08    #0.08
    err = 0.1
    #steering = (kp*steering)+(ki*steering*steering*0.5)+kd
    steering = (steering-err)*kp + (((steering*steering)/2)+err*steering)*ki+kd

    throttle = 1.0 - (speed / maxSpeed)
    save_throttle = throttle
    err_t = 1-throttle
    kp_t = 1
    ki_t = 0.001
    kd_t = 0.08
    throttle = (throttle-err_t)*kp_t+(((throttle*throttle)/2)+err_t*throttle)*ki_t+kd_t
    if throttle>0.99:
        throttle = save_throttle
        #speed = 6.0
    print('{} {} {}'.format(steering, throttle, speed))
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    #model2 = load_model('modelHillTrack.h5')
    app = socketio.Middleware(sio, app)
    #print(app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
