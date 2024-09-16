# Food Vision Mini

## Project Overview
Food Vision Mini is a deep learning project that classifies images of food into three categories: pizza, steak, and sushi. This project demonstrates the implementation of a custom CNN model (TinyVGG) and explores the impact of data augmentation on model performance.

## Key Features
- Custom dataset preparation using PyTorch's ImageFolder
- Implementation of a custom CNN architecture (TinyVGG)
- Data augmentation techniques using torchvision transforms
- Model training and evaluation
- Comparison of model performance with and without data augmentation
- Custom image prediction functionality

## Dependencies
- PyTorch
- torchvision
- matplotlib
- pandas
- requests
- tqdm
- torchinfo

## Project Structure
1. Data Preparation
2. Model Architecture (TinyVGG)
3. Training and Evaluation
4. Results Comparison
5. Custom Image Prediction

## Custom Dataset Preparation (Important!)
The project emphasizes the importance of proper dataset preparation. Here are the key steps:

1. **Data Download and Extraction**: 
   - The script automatically downloads and extracts the dataset if not present.
   - Uses `requests` to download and `zipfile` to extract.

2. **Custom Dataset Class**:
   - Implements `ImageFolderCustom`, a custom dataset class inheriting from `torch.utils.data.Dataset`.
   - Allows for flexible data loading and transformation.

3. **Data Transforms**:
   - Utilizes `torchvision.transforms` for data augmentation.
   - Implements different transform pipelines for training and testing data.
   - Uses `TrivialAugmentWide` for advanced data augmentation.

4. **DataLoader Creation**:
   - Uses `torch.utils.data.DataLoader` for efficient batch processing.
   - Implements separate dataloaders for training and testing.

5. **Class Handling**:
   - Automatically detects and handles class names and indices.
   - Uses the `find_classes` function to map class names to indices.

## Model Architecture
The project uses a custom CNN architecture called TinyVGG, which consists of:
- Two convolutional blocks (each with two conv layers, ReLU activation, and max pooling)
- A classifier block (flatten layer and a linear layer)

## Training and Evaluation
- Implements custom `train_step` and `test_step` functions.
- Uses CrossEntropyLoss and Adam optimizer.
- Trains for a specified number of epochs.
- Records and plots training/testing loss and accuracy.

## Results Comparison
- Compares model performance with and without data augmentation.
- Uses pandas for data manipulation and matplotlib for visualization.

## Custom Image Prediction
- Implements a `pred_and_plot_image` function for making predictions on custom images.
- Handles image preprocessing and model inference.

## Usage
1. Run the script to download and prepare the dataset.
2. Train the model with and without data augmentation.
3. Compare the results using the provided visualization tools.
4. Use the `pred_and_plot_image` function to make predictions on custom images.

## Future Improvements
- Experiment with more complex architectures.
- Implement transfer learning using pre-trained models.
- Expand the dataset to include more food categories.
