Uses --save model

This model performs 640 experiments on TI over 32 different parameter settings and 20 seeds

Changes specifically are to add frozen weights and pretrained weights

This Model.TI uses 2 channels


### Logic 

Mnist_TI.py
* Makes the ordering to turn ranks into digits
* Takes in the exception pair ranks

Datasets
* Takes in the ordering
* Converts the exception pair ranks to the actual digits from the ordering
