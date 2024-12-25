# blood-vessel-segmentation-using-deep-neural-network

Retinal Blood Vessel Segmentation Project
This project focuses on automating the segmentation of retinal blood vessels from fundus images using a U-Net-based deep learning model.
The pipeline includes preprocessing, training, testing, evaluation, and custom metrics for performance monitoring.

Workflow Overview

1. Image Preprocessing and Data Augmentation
Preprocessing:
Resized images to a fixed dimension (512x512 pixels).
Normalized pixel values to the range [0, 1] for consistent model input.
Masks were also resized and normalized for use as ground truth labels.
Data Augmentation:
Applied techniques like rotation, flipping, and scaling.
Increased dataset variability to improve model generalization.


2. Model Architecture (model.py)
Implemented a U-Net architecture, widely used for image segmentation.
Key Features:
Encoder: Extracts features by downsampling the image.
Decoder: Reconstructs the image by upsampling and using skip connections.
Skip Connections: Merge encoder and decoder features to retain details.
Output Layer:
A single-channel output (grayscale mask) with sigmoid activation.
Provides pixel-wise probabilities for segmentation.


3. Training Process (training.py)
Used preprocessed and augmented datasets for training.
Key Components:
Loss Function: Custom Dice Loss to maximize overlap.
Metrics: Dice Coefficient, IoU, Precision, and Recall.
Optimization: Adam optimizer with a learning rate scheduler.
Training Pipeline:
Loaded and shuffled data.
Created TensorFlow datasets for efficient batching and prefetching.
Incorporated callbacks like:
ModelCheckpoint: Save the best model.
ReduceLROnPlateau: Adjust learning rate when validation loss plateaus.
EarlyStopping: Stop training when performance stops improving.


4. Testing and Prediction (testing.py)
Evaluated the model's performance on unseen test data.
Steps:
Loaded the trained model and test images.
Predicted segmentation masks for test images.
Compared predictions with ground truth masks to calculate metrics.
Saved results, including:
Visual comparisons of input images, ground truth, and predictions.
Metrics for each test image in a CSV file.


5. Evaluation (eval.py)
Tested individual images for quick inference and validation.
Steps:
Preprocessed the input image to match model input size.
Predicted the segmentation mask using the trained model.
Output:
Raw segmentation mask.
Saved or displayed results for inspection.


6. Custom Metrics (metrics.py)
Defined metrics tailored for image segmentation:
Dice Coefficient: Measures overlap between prediction and ground truth.
IoU (Intersection over Union): Quantifies segmentation accuracy.
Dice Loss: Derived from Dice Coefficient to optimize training.


7. Key Features
Pipeline:
End-to-end process covering preprocessing, training, testing, and evaluation.
Generalization:
Used data augmentation to improve robustness on diverse images.
Metrics:
Comprehensive metrics for thorough performance analysis.
Model Saving:
Automatically saved the best-performing model for deployment.


8. Usage Instructions
Preprocess Images:
Use preprocessing scripts to resize and normalize the dataset.
Train the Model:
Run training.py to train the U-Net model.
Test the Model:
Use testing.py to evaluate the model on test data.
Evaluate Single Images:
Use eval.py for inference on individual images.
Inspect Metrics:
Review metrics and CSV logs for model performance analysis.


9. Dependencies
Python
TensorFlow/Keras
OpenCV
NumPy
Pandas
scikit-learn
tqdm

10. Conclusion
This project demonstrates a complete deep learning pipeline for retinal blood vessel segmentation.
It emphasizes preprocessing, data augmentation, and advanced evaluation metrics, making it robust for medical imaging tasks.
