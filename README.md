This project implements a Convolutional Neural Network (CNN) to classify medical images for the presence of Inflammatory Bowel Disease (IBD). 
The model achieves 92% validation accuracy after 10 training epochs.

Goal: Detect whether an input image indicates IBD or a healthy state.

Architecture: 3 convolutional layers + fully connected layers.

Accuracy: ~92% on validation data.

Batch size: 32 for training, 64 for validation.

Epochs: 10

Frameworks: PyTorch, Torchvision

This model was trained using images from the HyperKvasir dataset, with an 80/20 split between training and validation folders respectively.

Training Loss vs. Epoch Plot:

<img width="960" height="720" alt="training_loss" src="https://github.com/user-attachments/assets/351a1935-8098-496c-b172-303585286440" />

Accuracy vs. Epoch Plot:

<img width="960" height="720" alt="accuracy_curves" src="https://github.com/user-attachments/assets/d852f88b-aef3-4eb0-a1e2-019ad41c94b6" />

