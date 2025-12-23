# -*- coding: utf-8 -*-

import os
import cv2

image_folder = '/home/zyl/ours1/mmtracking/videos'
video_output = f'{image_folder}/videos.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

images.sort(key=lambda x: int(x.split('.')[0]))

first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_output, fourcc, 15, (width, height))

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()

print(f"Video saved at {video_output}")