import Augmentor
import os
import random
import shutil
import cv2

# A lot of functions like blur detection is not integrated. so check thoroughly.
def preprocess_images(input_folder):

    # Load images from input folder
    p = Augmentor.Pipeline(input_folder)

    # Add augmentation operations    
    p.flip_left_right(probability=0.5)
    p.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5)
    p.random_color(probability=0.5, min_factor=0.8, max_factor=1.2)
    p.zoom_random(probability=0.5, percentage_area=0.9)
    p.gaussian_distortion(probability=0.5, grid_width=7, grid_height=7, magnitude=5, corner='bell',method="in")
    p.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)

    # Choose 30% of images to augment
    image_list = os.listdir(input_folder)
    num_images = len(image_list)
    image_indices = random.sample(range(num_images), int(num_images*0.3))

    # Apply augmentation on chosen images
    p.sample(len(image_indices), multi_threaded=True)

    # Save augmented images to new folder
    p.process()

    # Get path of current Python file
    curr_file = os.path.abspath(__file__)
    curr_dir = os.path.dirname(curr_file)
    
    # Move original images up one level
    input_images = os.path.join(input_folder, 'output')
    new_dir = os.path.join(curr_dir, 'processed images')
    shutil.move(input_images, new_dir)

# Example usage for augmentation
training_input_folder = r'Attendance system\training data'
preprocess_images(training_input_folder)




