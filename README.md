# Propeller Machine Learning Surrogate Model

This project attempts to make a Pytorch based learned surrogate model to predict performance for the range of APC propellers.

The data, data loading script, training, and usage code is presented as is. Model weights for a 4x128x128x3 network after 1,000 epochs is also provided.

Since this program is intended to be used to select between different APC propeller offerings, it is likely very overfit and accuracy on propellers outside of the dataset is not guaranteed. For optimization purposes, be sure to restrict the solution to within the range of propellers in the dataset, shown in the graphic below.
