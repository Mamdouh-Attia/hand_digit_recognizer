import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from skimage import data, color, feature

class Processing:
    dataset_dir=""
    def __init__(self,dataset_dir):  
        self.dataset_dir=dataset_dir

    def preprocess_image(self,image):
        resized_image = cv2.resize(image, (128, 64))
        ycrcb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YCrCb)

        # Convert the input image to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Split channels
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)

        # Apply CLAHE to the Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_channel = clahe.apply(y_channel)

        # Merge channels back together
        enhanced_ycrcb = cv2.merge((y_channel, cr_channel, cb_channel))

        lower_skin = (0, 135, 85)
        upper_skin = (255, 180, 135)

        # Apply the mask
        mask = cv2.inRange(enhanced_ycrcb, lower_skin, upper_skin)

        # Define the kernel size
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Apply erosion to remove noise
        mask = cv2.erode(mask, kernel, iterations=1)

        # Apply dilation to fill gaps
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fill the largest contour (assumed to be the hand)
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [max_contour], 0, (255), -1)

        # Apply erosion and dilation
        kernel_size = 7
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Apply the mask to the grayscale image
        masked_gray_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)

        # Apply a bilateral filter to reduce noise while preserving edges
        denoised_image = cv2.GaussianBlur(masked_gray_image, (5, 5), 0)

        return denoised_image
    
    def rotate_image(self,image, angle):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image
    def ReadAndProcess(self,rotation_angles):
        data_train = []
        labels = []
        for folder in os.listdir(self.dataset_dir):
            for img_path in os.listdir(self.dataset_dir + '/' + folder):
                img = cv2.imread(self.dataset_dir + '/' + folder + '/' + img_path, -1)
                if img is None:
                    print(f'Image could not be read: {self.dataset_dir}/{folder}/{img_path}')
                else:
                    preprocessed_img = self.preprocess_image(img)

                    preprocessed_img = self.preprocess_image(img)

                    # Add the original preprocessed image and label to the data_train and labels lists
                    data_train.append(preprocessed_img)
                    labels.append(int(folder))

                    # Create rotated versions of the preprocessed image and add them to the data_train and labels lists
                    for angle in rotation_angles:
                        rotated_image = self.rotate_image(preprocessed_img, angle)
                        data_train.append(rotated_image)
                        labels.append(int(folder))
        return data_train,labels

    def ShowSomeImages(self,axes,data_train,labels,random_indices):
        for i, ax in enumerate(axes.flatten()):
            # Specify 'gray' colormap for grayscale images
            ax.imshow(data_train[random_indices[i]], cmap='gray')
            ax.set_title(f'Label: {labels[random_indices[i]]}')
            ax.axis('off')
            plt.tight_layout()
            plt.show()

