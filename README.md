Pytorch implementation of Relational Networks - [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf)

Implemented & tested on Sort-of-CLEVR task. Augmented with support for state description task on the Sort-of-CLEVR dataset.

## Sort-of-CLEVR

Sort-of-CLEVR is simplified version of [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/).This is composed of 10000 images and 20 questions (10 relational questions and 10 non-relational questions) per each image. 6 colors (red, green, blue, orange, gray, yellow) are assigned to randomly chosen shape (square or circle), and placed in a image.

Non-relational questions are composed of 3 subtypes:

1) Shape of certain colored object
2) Horizontal location of certain colored object : whether it is on the left side of the image or right side of the image
3) Vertical location of certain colored object : whether it is on the upside of the image or downside of the image

Theses questions are "non-relational" because the agent only need to focus on certain object.

Relational questions are composed of 3 subtypes:

1) Shape of the object which is closest to the certain colored object
1) Shape of the object which is furthest to the certain colored object
3) Number of objects which have the same shape with the certain colored object

These questions are "relational" because the agent has to consider the relations between objects.

Questions are encoded into a vector of size of 11 : 6 for one-hot vector for certain color among 6 colors, 2 for one-hot vector of relational/non-relational questions. 3 for one-hot vector of 3 subtypes.

<img src="./data/sample.png" width="256">

I.e., with the sample image shown, we can generate non-relational questions like:

1) What is the shape of the red object? => Circle (even though it does not really look like "circle"...)
2) Is green object placed on the left side of the image? => yes
3) Is orange object placed on the upside of the image? => no

And relational questions:

1) What is the shape of the object closest to the red object? => square
2) What is the shape of the object furthest to the orange object? => circle
3) How many objects have same shape with the blue object? => 3

## Requirements

- Python 
- [numpy](http://www.numpy.org/)
- [pytorch](http://pytorch.org/)
- [opencv](http://opencv.org/)

## Usage
The code can be executed directly from the notebook included above on Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mascinissa/relational-networks/blob/master/Sort_of_CLEVR_RN_notebook.ipynb)

You can also run the scripts separately, as follows:

### Images version of Sort-of-CLEVR task

  	$ python SoCLEVR_images_generator.py

to generate the images version of the sort-of-clevr dataset
and

 	 $ python train_SoCLEVR_images.py

to train the model.

### State description version of Sort-of-CLEVR task

  	$ python SoCLEVR_state_description_generator.py

to generate the state description version of the sort-of-clevr dataset
and

 	 $ python train_SoCLEVR_state_description.py

to train the model.

## Result

| | Images version (40th epoch) | State description version (25th epoch)|
| --- | --- | --- |
| Non-relational question | 99% | 99% |
| Relational question | 89% | 96% |

