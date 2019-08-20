from matplotlib import pyplot as plt
import os
import cv2

new_images_path = './My_Images'

test_X = []
test_Y = []

for image_file in os.listdir(new_images_path):
    image = mpimg.imread('/'.join([new_images_path, image_file]))
    image_type = image_file.split('.')[0]
    resized = cv2.resize(image, (32, 32))
    test_X.append(resized)
    test_Y.append(int(image_type))

test_X = np.array(test_X, dtype=np.float32)
test_Y = np.array(test_Y, dtype=np.int32)

fig, plts = plt.subplots(6)
num = 0
for image_file in os.listdir(new_images_path):
    image = plt.imread('/'.join([new_images_path, image_file]))
    image_type = image_file.split('.')[0]
    resized = cv2.resize(image, (32, 32))
    plts[num].imshow(resized)
    print(num)
    num += 1
plt.show()