"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from time import sleep

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import cv2
from sense_emu import SenseHat


#initialize SenseHat instance and clear the LED matrix
sense = SenseHat()
sense.clear()

#Raspimon RGB colors
r = [139, 0, 0]
mon_w = [255, 255, 255]
mon_y = [250, 214, 29]
mon_oj = [225, 151, 32]
mon_br = [129, 30, 9]
mon_r = [246, 45, 20]
mon_b = [0, 0, 0]
mon_p = [255,105,180]
sky_b = [135,206,235]
grass_g = [0,154,23]

#Raspimon Idle
pimon_idle = [
    sky_b, mon_b, mon_b, sky_b, sky_b, sky_b, sky_b, mon_b,
    sky_b, sky_b, mon_y, mon_oj, sky_b, sky_b, sky_b, mon_oj,
    sky_b, sky_b, sky_b, mon_y, mon_y, mon_y, mon_y, mon_oj,
    mon_oj, mon_oj, sky_b, mon_y, mon_b, mon_y, mon_y, mon_b,
    mon_oj, mon_oj, sky_b, mon_r, mon_y, mon_y, mon_y, mon_oj,
    sky_b, mon_br, sky_b, mon_y, mon_oj, mon_oj, mon_oj, sky_b,
    grass_g, mon_br, mon_y, mon_oj, mon_y, mon_oj, mon_y, grass_g,
    grass_g, grass_g, mon_y, mon_oj, mon_br, mon_br, mon_oj, grass_g
]

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '-i',
      '--image',
      default='tmp/grace_hopper.bmp',
      help='image to be classified')
    parser.add_argument(
      '-m',
      '--model_file',
      default='tmp/mobilenet_v1_1.0_224.tflite',
      help='.tflite model to be executed')
    parser.add_argument(
      '-l',
      '--label_file',
      default='tmp/labels.txt',
      help='name of file containing labels')
    parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
    parser.add_argument(
      '--input_std',
      default=127.5, type=float,
      help='input standard deviation')
    parser.add_argument(
      '--num_threads', default=None, type=int, help='number of threads')
    args = parser.parse_args()

    interpreter = tflite.Interpreter(
      model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    sense.set_pixels(pimon_idle)
    #capture video
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        file = 'frame.png'
        cv2.imwrite(file,frame)

        # Display the resulting frame
        cv2.imshow('frame',frame) #show camera output
        key = cv2.waitKey(0) #press 0 to move through frames
        if key == ord('q'): #press q to quit
            break
        elif key == ord('y'):
            img = Image.open(file).resize((width, height))

            # add N dim
            input_data = np.expand_dims(img, axis=0)

            if floating_model:
                input_data = (np.float32(input_data) - args.input_mean) / args.input_std

            interpreter.set_tensor(input_details[0]['index'], input_data)

            start_time = time.time()
            interpreter.invoke()
            stop_time = time.time()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            results = np.squeeze(output_data)

            #top_k = results.argsort()[-5:][::-1]
            top_k = results.argsort()[-1:]
            labels = load_labels(args.label_file)
            
            for i in top_k:
                if floating_model:
                    label = labels[i]
                    obj = label[label.index(':') + 1:]
                    print(obj)
                    sense.show_message('This is a...', text_colour=mon_y, scroll_speed=0.05)
                    sense.show_message(obj, text_colour=mon_r, scroll_speed=0.1)
                    sleep(1)
                    sense.set_pixels(pimon_idle)
    
cap.release()
cv2.destroyAllWindows()
