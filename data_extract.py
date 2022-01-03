# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:34:24 2022

@author: akhil
"""
import tensorflow as tf

data = tf.keras.utils.image_dataset_from_directory(
    "./data/extracted_images/1",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
)