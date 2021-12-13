import os
import glob

image_dir = "./data/images/"
frames = os.listdir(image_dir)
frames.sort()
frames = [f.replace(".jpg", "") + '\n' for f in frames]
# labels.sort()

with open("list.txt", "w") as f:
    f.writelines(frames)
