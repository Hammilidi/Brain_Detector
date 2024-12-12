# Brain Tumor Classification with VGG16

This project aims to classify brain tumor images using a convolutional neural network (CNN) based on the VGG16 architecture. The model is fine-tuned to classify images as either containing a brain tumor or not. The project is built using TensorFlow and Keras, and includes an interactive Streamlit app for real-time image prediction.

## Project Structure

The repository contains two main scripts:

1. **`train_model.py`**: 
   - Trains a binary classification model using the VGG16 architecture pre-trained on ImageNet.
   - The model is fine-tuned with custom layers for binary classification.
   - Includes data preprocessing, model compilation, and training using data augmentation.
   - Saves the trained model for future use.

2. **`app.py`**:
   - A Streamlit application that allows users to upload an image for prediction.
   - The app loads the pre-trained model and predicts whether the uploaded image contains a brain tumor or not.

## Requirements

- Python 3.6 or later
- TensorFlow 2.x
- Streamlit
- Matplotlib
- Numpy

To install the necessary libraries, you can use `pip`:

```bash
pip install -r requirements.txt
```

## Data

The dataset used for training is assumed to be in the following structure:

```
brain_tumor_dataset/
    train/
        tumor/
        no_tumor/
    val/
        tumor/
        no_tumor/
```

Make sure to replace `brain_tumor_dataset` with the actual path to your dataset.

## How to Train the Model

To train the model, run the following command in your terminal:

```bash
python train_model.py
```

This script will:
1. Load the pre-trained VGG16 model (without the top classification layer).
2. Add custom layers for binary classification.
3. Compile and train the model on the dataset with data augmentation.
4. Save the trained model as `brain_tumor_vgg_model.keras`.

## How to Use the Streamlit App

To run the Streamlit app, use the following command:

```bash
streamlit run app.py
```

This will start a local Streamlit server and open the app in your default browser. You can upload an image (in `.jpg`, `.jpeg`, or `.png` format), and the model will predict whether it shows a brain tumor or not.

## Example of Image Prediction

In the Streamlit app, after uploading an image, the model will display the image and predict:

- **Tumor** if the image contains a tumor.
- **No Tumor** if the image doesn't contain a tumor.

## Model Overview

The model is built using the VGG16 architecture with the following modifications:
- The pre-trained VGG16 model is used as the base (excluding the top layer).
- The model is extended with a `Flatten` layer, a dense layer with 128 neurons, a dropout layer for regularization, and a final dense layer with a sigmoid activation for binary classification.

## Notes

- The dataset used for training must be organized with two folders: `tumor` and `no_tumor` under `train` and `val` directories for training and validation data, respectively.
- The model was trained with a learning rate of `0.0001` and an early stopping callback to prevent overfitting.

## License

This project is licensed under the MIT License.

---

### Ajouter un fichier `requirements.txt`

Si vous ne l'avez pas encore, voici un fichier `requirements.txt` pour installer les dépendances :

```
tensorflow==2.11.0
streamlit==1.15.0
matplotlib==3.7.0
numpy==1.24.0
```

---

Ce README couvre toutes les étapes nécessaires pour comprendre, utiliser et entraîner le modèle, tout en fournissant des instructions pour démarrer avec l'application Streamlit. Vous pouvez l'adapter si nécessaire en fonction de vos spécificités.
