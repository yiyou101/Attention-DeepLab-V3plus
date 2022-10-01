A deep learning approach to extract the glacier outlines
====
This repository contains code for extracting the glacier outlines in high spatial resolution remote sensing using deep learning. 

For an explanation of the concepts, ideas and evaluation, see publication below. Please use the following citation when using the code for a publication:

Chu, X., Yao, X., Duan, H., Chen, C., Li, J., and Pang, W.: Glacier extraction based on high spatial resolution remote sensing images using a deep learning approach with attention mechanism, The Cryosphere. https://doi.org/10.5194/tc-2022-61, 2022.

More documentation is provided in the form of docstrings throughout the code.

### dataset:

The sample datasets of glacier are as follow:

![image](https://user-images.githubusercontent.com/82889935/190320208-8652b4c8-7aa8-42f2-882a-671450248777.png)

[download Baidu Drive](https://pan.baidu.com/s/1P0FFkq3zrIbYfDVTLC_soA?pwd=ctsa )

More documentation is provided in the form of docstrings throughout the code.

### Structure

***utils**: The code for reading satellite images and ground truth labels in the form of pytorch datasets. This includes code for loss functions, data augmentation and Accuracy evaluation.

***model**: The attention DeepLab V3+ model architecture and parts.

***train**: The code for traing the model and validation the model.

***predict**: The code for test the performance of the model and predict the result by add the test time augmentation strategy.
