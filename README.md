# equation-solver
This is a machine learning algorithm which processes an image of an equation and solves it

#about dataset
the original dataset consists of large number of images of numbers and signs(dataset link-https://www.kaggle.com/xainano/handwrittenmathsymbols). They all have a color code of rgb. We later convert these images into a csv file("dataset.csv") in order
for our model to process them

#data preprocessing
In order for our model to process the data, we need to convert the images into a csv file. The code for doing this is available in data_extraction.ipynb file. we loop thorugh each directory
and image, grayscale them and normalize them. We also resize the images to 96 x 96 pixels. Then we append these processed images into a list and then we convert this list into a pd
dataframe

#model building
for this project I used convolutional neural networks. I used keras sequential api to build the convolutional neural networks. In order to optimize the algorithm I used a tool called optuna
