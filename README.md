# MRI Brain Tumor Classification

This project builds a deep learning model to classify brain MRI scans into four categories:

- Glioma
- Meningioma
- No tumor
- Pituitary tumor

The workflow is implemented in the notebook `Brain_Tumor_Cassification_Machine_LearningProject_.ipynb` and uses a transfer-learning approach based on EfficientNetB0, a pretrained convolutional neural network from ImageNet.

## Project Goal

The goal of this project is to automate brain tumor detection from MRI images and support clinical-style decision making by identifying the most likely tumor class from an image.

This is a supervised image classification task using labeled MRI images and a deep learning pipeline that includes:

- dataset loading and preprocessing
- class distribution analysis
- train/validation/test splitting
- data augmentation
- transfer learning with a pretrained CNN
- model tuning and early stopping
- evaluation using classification metrics and confusion matrices
- Grad-CAM-style visual explanations for sample predictions
- model saving and reusable prediction utilities

## Dataset

The project downloads the dataset from Kaggle using `kagglehub`:

- Source: `tombackert/brain-tumor-mri-data`
- Folder used: `brain-tumor-mri-dataset`
- Classes: `glioma`, `meningioma`, `notumor`, `pituitary`

The notebook scans all image files in each class directory, validates image readability, and builds a DataFrame containing:

- file path
- corresponding class label

## Methodology

### 1. Data preparation

The notebook:

- collects all image files from the four MRI class folders
- checks for missing or corrupted images
- looks at image sizes and distributions
- splits data into training, validation, and test sets with stratification
- visualizes class balance before training

### 2. Data augmentation

To improve model robustness, the training pipeline uses `ImageDataGenerator` with augmentation such as:

- rotation
- width and height shifts
- zoom
- horizontal flipping

Validation and test images are kept in their original form to evaluate the model fairly.

### 3. Model architecture

The model uses a pretrained `EfficientNetB0` backbone:

- `weights="imagenet"`
- `include_top=False`
- input size: `224 x 224 x 3`

After the backbone, the notebook adds:

- GlobalAveragePooling2D
- Dropout layers
- a dense hidden layer with ReLU activation
- softmax output layer for the four classes

### 4. Training strategy

The project applies a two-phase transfer learning process:

1. Train the classification head with the base model frozen
2. Fine-tune selected layers of EfficientNetB0 with a lower learning rate

It also uses callbacks for:

- model checkpointing
- early stopping
- learning-rate reduction on plateau

## Evaluation

The notebook evaluates the model using standard classification tools such as:

- confusion matrix
- classification report
- F1 score
- validation accuracy and loss curves
- visual inspection of predictions

It also includes Grad-CAM-style interpretability checks to inspect which parts of an MRI image contribute most to a prediction.

## Files in this Repository

- `README.md` — project overview and instructions
- `Brain_Tumor_Cassification_Machine_LearningProject_.ipynb` — full end-to-end training and evaluation workflow

## Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn
- pandas
- NumPy
- matplotlib
- seaborn
- Pillow
- KaggleHub

## Setup

Create a Python environment and install the required libraries:

```bash
pip install tensorflow pandas numpy matplotlib seaborn pillow scikit-learn kagglehub
```

Then open the notebook and run the cells in order.

## Usage

1. Open `Brain_Tumor_Cassification_Machine_LearningProject_.ipynb` in Jupyter or VS Code.
2. Run the notebook cells to download the dataset, prepare the data, train the model, and evaluate it.
3. The notebook saves the best model as `best_brain_tumor_model.keras` and then exports it as `final_brain_tumor_model.keras`.
4. Use the included prediction helper to classify new MRI images.

## Example Prediction Flow

The project includes a reusable function named `predict_mri(...)` that:

- loads an image
- resizes it to the model input size
- runs inference
- returns the predicted class and probability output

## Notes

This project is a strong example of transfer learning for medical image classification. It demonstrates how to combine preprocessing, augmentation, pretrained CNNs, and explainability techniques into a practical machine learning pipeline.

## License

This project is provided under the repository license included in the project files.
