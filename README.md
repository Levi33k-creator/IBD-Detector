This project implements a Convolutional Neural Network (CNN) to classify medical images for the presence of Inflammatory Bowel Disease (IBD). 
The model achieves 92% validation accuracy after 10 training epochs.

Goal: Detect whether an input image indicates IBD or a healthy state.

Architecture: 3 convolutional layers + fully connected layers.

Accuracy: ~92% on validation data.

Batch size: 32 for training, 64 for validation.

Epochs: 10

Frameworks: PyTorch, Torchvision

This model was trained using images from the HyperKvasir dataset, with an 80/20 split between training and validation folders respectively.

