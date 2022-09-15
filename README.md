A deep learning approach to extract the glacier outlines
====
This repository contains code for extracting the glacier outlines in high spatial resolution remote sensing using deep learning. 

### dataset:

The sample datasets of glacier are as follow:

![image](https://user-images.githubusercontent.com/82889935/190320208-8652b4c8-7aa8-42f2-882a-671450248777.png)

[download Baidu Drive](https://pan.baidu.com/s/1WUGkOzeAS1kwPoe991RfWA?pwd=23tr)

More documentation is provided in the form of docstrings throughout the code.

### Structure

* **utils**: code for reading satellite images and ground truth labels in the form of pytorch datasets
* **evaluation**: various scripts for running quantitative and qualitative evaluation
* **experiments**: pytorch lightning module for running experiments in an easily configurable way
* **modeling**: model architecture and parts
* **trainer**: contains training scripts
* **utils**: various utilities for data manipulation, visualization, etc.
