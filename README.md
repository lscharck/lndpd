# lndpd
  This serves as the official repository for the landing pad project. Dubbed "lndpd" this project aims to develop a camera integrated landing pad detection system for a drone. Currently deep learning is being investigated as the intelligence behind the detection system. Pytorch is the deep learning library of choice preforming all AI task for this project.
  
The directory structure is as follows:
datagen - holds all code related to generating training data and validation data for the CNN
pytorch - holds all code related to training and inferencing the CNN
  
The code in datagen references several external directories that hold the training and validation data. Additionally some of the code requires a file at the top level of datagen directory for the "real data" labels.

The code in pytorch is self-contained and requires no external directories.
