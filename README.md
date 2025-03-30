# Propeller Machine Learning Surrogate Model

This project attempts to make a Pytorch based learned surrogate model to predict performance for the range of APC propellers.

The data, data loading script, training, and usage code is presented as is. Model weights for a 4x128x128x3 network after 1,000 epochs is also provided.

Since this program is intended to be used to select between different APC propeller offerings, it is likely very overfit and accuracy on propellers outside of the dataset is not guaranteed. For optimization purposes, be sure to restrict the solution to within the range of propellers in the dataset, shown in the graphic below.

![image](https://github.com/user-attachments/assets/39ba6098-399c-4602-baea-4c5f8a38ce87)

# Results

![image](https://github.com/user-attachments/assets/435d51fc-8b32-47e4-a1c0-69f615122a52)
![image](https://github.com/user-attachments/assets/fde7d6cf-12dd-4eee-bd54-68dbebf144e3)
![image](https://github.com/user-attachments/assets/c1cdd852-eac3-4b86-9c12-0eab5fe69228)
