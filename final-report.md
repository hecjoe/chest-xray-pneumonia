# Chest X-Ray Images (Pneumonia) - Final Report

**Team Members:** Jesus Villa, Derek Shin, Momoka Aung, Gauri Chahal, Joshua Encinas  
**Project Lead:** Jesus Villa  
**Department:** Department of Computer Science, California State University Los Angeles  
**Course:** CS4662 - Advanced Machine & Deep Learning  
**Date:** May 10, 2026

---

# Abstract

This project focuses on developing a machine learning model that classifies chest X-ray images as either **NORMAL** or **PNEUMONIA**. The goal is to explore how computer vision and deep learning can be applied to medical image classification, especially for detecting pneumonia from radiographic images. The dataset used for this project is the Chest X-Ray Images (Pneumonia) dataset from Kaggle, which contains thousands of labeled JPEG chest X-ray images split into training, validation, and testing folders.

The implemented model is a baseline Convolutional Neural Network (CNN). The pipeline loads grayscale X-ray images, resizes them to 128 x 128 pixels, normalizes pixel values, trains the CNN, and evaluates the model using accuracy, precision, recall, F1-score, a confusion matrix, and a ROC curve with AUC. Based on the provided final run, the CNN achieved approximately **76% test accuracy** and a **ROC-AUC of 0.93**. The model performed especially well at detecting pneumonia cases, with pneumonia recall of **0.99**, but it also produced many false positives by classifying many normal X-rays as pneumonia.

---

# 1. Introduction

## 1.1 Background

Pneumonia is a lung infection that can cause inflammation and fluid buildup in the air sacs of the lungs. Chest X-rays are commonly used by medical professionals to help identify pneumonia-related abnormalities. However, interpreting medical images can be time-consuming and may vary depending on clinical experience. Machine learning, especially deep learning, can help support medical image classification by learning visual patterns from labeled examples.

This project applies computer vision techniques to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**. The project is intended as an educational machine learning application and is not meant to replace professional medical diagnosis.

## 1.2 Problem Statement

The problem addressed in this project is binary image classification. Given a chest X-ray image, the model must predict whether the image belongs to the **NORMAL** class or the **PNEUMONIA** class. The main challenge is training a model that can correctly identify pneumonia cases while also avoiding unnecessary false positives for normal patients.

## 1.3 Objectives

The main objectives of the project are:

1. Load and preprocess a labeled chest X-ray image dataset.
2. Build a CNN model for binary classification.
3. Train the model using the training and validation data.
4. Evaluate the model on the test set using accuracy, precision, recall, F1-score, a confusion matrix, and ROC-AUC.
5. Interpret the model's strengths and weaknesses based on the final results.
6. Identify missing requirements and future improvements, including the need for multiple model comparisons.

## 1.4 Contributions

The project contribution is a working deep learning pipeline for chest X-ray pneumonia classification. The completed code includes modules for data loading, preprocessing, CNN model construction, baseline training configuration, and model evaluation. The project also produces visualizations that help explain training behavior and test performance, including learning curves, a confusion matrix, classification metrics, and a ROC curve.

---

# 2. Dataset Overview

## 2.1 Dataset Description

The project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle. The dataset contains labeled JPEG chest X-ray images divided into two classes:

- **NORMAL:** chest X-rays without pneumonia.
- **PNEUMONIA:** chest X-rays showing pneumonia-related findings.

The dataset is pre-organized into training, validation, and test folders. The Kaggle project page and repository description list the dataset as approximately 5,863 images. In the commonly extracted folder structure used for the run, the dataset contains 5,856 image files across the train, validation, and test folders.

## 2.2 Dataset Structure

The dataset folder is organized as follows:

```text
chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

## 2.3 Class Distribution

The dataset is imbalanced because the pneumonia class contains more images than the normal class. The typical extracted split is shown below:

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

This imbalance is important because the model may learn to favor the pneumonia class. The test results show that this became a major issue: the CNN detected nearly all pneumonia cases, but it incorrectly labeled many normal X-rays as pneumonia.

## 2.4 Challenges in the Dataset

The main dataset challenges are:

- **Class imbalance:** The pneumonia class has many more images than the normal class.
- **Small validation set:** The validation folder contains only 16 images, so validation accuracy can fluctuate sharply from epoch to epoch.
- **Image variability:** X-rays can differ in brightness, contrast, positioning, and patient anatomy.
- **Medical image complexity:** Pneumonia patterns can be subtle, and normal images may contain visual features that confuse the model.
- **Risk of overfitting:** The CNN can learn the training images very well while still struggling to generalize to normal test images.

---

# 3. Data Preprocessing

## 3.1 Image Loading

Images are loaded using TensorFlow/Keras image utilities. The preprocessing code loops through the `NORMAL` and `PNEUMONIA` folders and assigns binary labels:

| Class | Label |
|---|---:|
| NORMAL | 0 |
| PNEUMONIA | 1 |

The code loads images from the train, validation, and test folders separately.

## 3.2 Image Resizing

All images are resized to **128 x 128 pixels**. This creates a consistent input size for the CNN and reduces computational cost compared with using full-resolution X-ray images.

## 3.3 Data Normalization

Pixel values are converted to floating-point values and normalized by dividing by 255. This changes the image pixel range from 0-255 to 0-1. Normalization helps the model train more efficiently and stabilizes gradient-based optimization.

## 3.4 Data Augmentation

The current final code currently loads, resizes, and normalizes the images without applying transformations such as rotation, zoom, shifting, or flipping.

## 3.5 Training, Validation, and Test Split

The project uses the dataset's original split:

- **Training set:** used to fit the CNN weights.
- **Validation set:** used to monitor model performance during training.
- **Test set:** used for final model evaluation.

Because the validation set is very small, the validation accuracy and validation loss curves should be interpreted carefully. A single misclassified validation image can cause a large change in the validation metrics.

---

# 4. Methodology

## 4.1 Model Architecture

The implemented model is a baseline Convolutional Neural Network built using TensorFlow/Keras. The CNN accepts grayscale chest X-ray images with shape:

```text
128 x 128 x 1
```

The model architecture is:

| Layer Type | Details | Purpose |
|---|---|---|
| Input | 128 x 128 x 1 grayscale image | Accept preprocessed X-ray image |
| Conv2D | 32 filters, 3 x 3 kernel, ReLU | Learn low-level image features |
| MaxPooling2D | 2 x 2 pool size | Reduce spatial dimensions |
| Conv2D | 64 filters, 3 x 3 kernel, ReLU | Learn deeper feature patterns |
| MaxPooling2D | 2 x 2 pool size | Reduce spatial dimensions |
| Conv2D | 128 filters, 3 x 3 kernel, ReLU | Learn higher-level image features |
| MaxPooling2D | 2 x 2 pool size | Reduce spatial dimensions |
| Flatten | Converts feature maps to vector | Prepare features for dense layers |
| Dense | 128 units, ReLU | Classification feature learning |
| Dropout | 0.5 | Reduce overfitting |
| Dense | 1 unit, sigmoid | Binary classification output |

### 4.1.1 Convolutional Layers

The convolutional layers extract visual patterns from the X-ray images. Early layers learn simple features such as edges and brightness changes, while deeper layers learn more complex structures that may help distinguish normal lungs from pneumonia-related abnormalities.

### 4.1.2 Pooling Layers

Max pooling layers reduce the size of the feature maps after each convolutional block. This decreases the number of parameters, improves computational efficiency, and helps the model focus on the strongest detected features.

### 4.1.3 Dense Layers

After the convolutional and pooling layers, the feature maps are flattened and passed into a dense layer with 128 units. This layer combines the extracted features and helps make the final classification decision.

### 4.1.4 Activation Functions

The convolutional and dense hidden layers use the **ReLU** activation function. The output layer uses a **sigmoid** activation function because the task is binary classification. A sigmoid output close to 0 represents **NORMAL**, while an output close to 1 represents **PNEUMONIA**.

### 4.1.5 Dropout Regularization

A dropout layer with rate **0.5** is used before the final output layer. Dropout randomly disables a portion of neurons during training, which helps reduce overfitting and improves generalization.

## 4.2 Training Configuration

### 4.2.1 Loss Function

The model uses **binary cross-entropy** as the loss function because the target variable has two classes.

### 4.2.2 Optimizer

The model uses the **Adam** optimizer. Adam is commonly used for deep learning because it adapts the learning rate during training and often performs well with limited manual tuning.

### 4.2.3 Batch Size

The baseline training configuration uses a batch size of **32**.

### 4.2.4 Number of Epochs

The number of epochs is set at 15.

## 4.3 Hyperparameter Tuning

Baseline parameters include:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Batch size | 32 |
| Epochs | 15 |
| Loss | Binary cross-entropy |


## 4.4 Implementation Tools

The project uses the following tools and libraries:

- **Python:** main programming language.
- **TensorFlow/Keras:** CNN model building, training, and image loading.
- **NumPy:** array processing.
- **Matplotlib:** result visualizations.
- **Scikit-learn:** classification report, confusion matrix, ROC curve, and AUC calculation.
- **Google Colab:** cloud notebook environment used to run the model and generate outputs.
- **Kaggle API:** used to download the dataset in Colab.

## 4.5 Code Organization

The repository is organized into separate Python modules:

| File | Purpose |
|---|---|
| `main.py` | Main pipeline that loads data, preprocesses images, builds the model, trains it, and evaluates it. |
| `preprocessing.py` | Loads images from folders, resizes them, converts them to grayscale arrays, and normalizes pixel values. |
| `model.py` | Defines and compiles the baseline CNN architecture. |
| `tuning.py` | Contains hyperparameter tuning logic and returns baseline parameters. |
| `evaluation.py` | Evaluates the trained model and generates classification metrics and visualizations. |

---

# 5. Experimental Results

## 5.1 Training Performance

The training accuracy curve showed that the CNN learned the training set quickly. Training accuracy started around **0.88** and increased steadily, approaching almost **1.00** by the final epochs. The training loss decreased from approximately **0.37** to about **0.02**, indicating that the model fit the training data very well.

This strong training performance shows that the CNN was able to learn patterns from the X-ray images. However, very high training accuracy can also indicate possible overfitting, especially when validation and test performance are less stable.

## 5.2 Validation Performance

The validation accuracy was generally high but fluctuated sharply between epochs. In the graphical plot, validation accuracy reached **1.00** during several epochs but also dropped at certain points, including near the final epoch. The validation loss also decreased overall but showed spikes during training.

These fluctuations are likely caused by the very small validation set. Since the validation folder contains only 16 images, each image has a large effect on validation accuracy and validation loss.

## 5.3 Test Performance

The model was evaluated on the test set containing **624 images**. The final classification report showed:

| Metric | Value |
|---|---:|
| Test Accuracy | 0.76 |
| ROC-AUC | 0.93 |
| Macro Avg Precision | 0.85 |
| Macro Avg Recall | 0.68 |
| Macro Avg F1-score | 0.69 |
| Weighted Avg Precision | 0.82 |
| Weighted Avg Recall | 0.76 |
| Weighted Avg F1-score | 0.73 |

The CNN achieved strong ROC-AUC but moderate accuracy. This means the model ranked the two classes well overall, but the default decision threshold produced many false positives for pneumonia.

## 5.4 Accuracy and Loss Curves

**Figure 1: Training and Validation Accuracy.** The training accuracy increased steadily across epochs. Validation accuracy was high but unstable, likely because of the small validation set.

**Figure 2: Training and Validation Loss.** Training loss decreased consistently. Validation loss decreased overall but showed instability and a final spike, suggesting possible overfitting or sensitivity to the small validation set.

## 5.5 Model Comparison Requirement Status

| Model | Status | Accuracy | AUC | Notes |
|---|---|---:|---:|---|
| Baseline CNN | Completed | 0.76 | 0.93 | Strong pneumonia recall, many false positives for pneumonia |
| Logistic Regression | Not completed | N/A | N/A | N/A |
| SVM | Not completed | N/A | N/A | Mentioned in project plan |
| Random Forest | Not completed | N/A | N/A | N/A |
| Transfer Learning CNN | Not completed | N/A | N/A | README mentions transfer learning |

---

# 6. Model Evaluation

## 6.1 Confusion Matrix

The confusion matrix from the provided visualization was:

| Actual Class | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| NORMAL | 84 | 150 |
| PNEUMONIA | 2 | 388 |

This means:

- **84** normal X-rays were correctly classified as normal.
- **150** normal X-rays were incorrectly classified as pneumonia.
- **388** pneumonia X-rays were correctly classified as pneumonia.
- **2** pneumonia X-rays were incorrectly classified as normal.

The model correctly classified:

```text
84 + 388 = 472 images
```

Out of:

```text
624 total test images
```

This gives an approximate accuracy of:

```text
472 / 624 = 0.756 = 75.6%
```

The most important observation is that the model almost never missed pneumonia cases, but it incorrectly flagged many normal cases as pneumonia.

## 6.2 Classification Report

The classification report from the final run showed:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.98 | 0.37 | 0.54 | 234 |
| PNEUMONIA | 0.73 | 0.99 | 0.84 | 390 |
| **Accuracy** |  |  | **0.76** | **624** |
| **Macro Avg** | **0.85** | **0.68** | **0.69** | **624** |
| **Weighted Avg** | **0.82** | **0.76** | **0.73** | **624** |

### 6.2.1 Precision

Precision measures how many predicted positives are actually correct. The **NORMAL precision of 0.98** means that when the model predicted an image was normal, it was usually correct. The **PNEUMONIA precision of 0.73** means that many images predicted as pneumonia were truly pneumonia, but a noticeable number were actually normal.

### 6.2.2 Recall

Recall measures how many actual examples of a class the model correctly identified. The **PNEUMONIA recall of 0.99** is very strong, meaning the model detected almost all pneumonia cases. The **NORMAL recall of 0.37** is weak, meaning the model missed many normal cases by labeling them as pneumonia.

### 6.2.3 F1-Score

F1-score balances precision and recall. The PNEUMONIA class had a higher F1-score of **0.84**, while the NORMAL class had a lower F1-score of **0.54**. This shows that the model performed much better on pneumonia detection than normal classification.

## 6.3 ROC Curve and AUC

The ROC curve showed an **AUC of 0.93**, which indicates strong overall separation between the two classes. A model with an AUC close to 1.00 is better at ranking positive and negative examples than a random classifier.

However, the confusion matrix shows that the selected decision threshold led to many false positives. This means the model has useful class separation ability, but the threshold may need to be adjusted to improve the balance between detecting pneumonia and correctly identifying normal X-rays.

## 6.4 Error Analysis

The model's main error pattern is over-predicting the pneumonia class. This can be seen in the confusion matrix, where **150 normal X-rays** were classified as pneumonia. This likely happened because of the class imbalance in the training set and because pneumonia images make up most of the dataset.

In a medical screening context, high pneumonia recall can be valuable because missing a pneumonia case could be more serious than incorrectly flagging a normal patient for follow-up. However, too many false positives can still create problems, such as unnecessary concern, extra testing, and inefficient use of medical resources.

## 6.5 Misclassified Samples


# 7. Discussion and Improvements

## 7.1 Strengths of the Model

The main strength of the CNN is its ability to detect pneumonia cases. The model achieved a pneumonia recall of **0.99**, meaning it correctly identified almost every pneumonia image in the test set. The ROC-AUC of **0.93** also suggests that the model learned meaningful visual differences between normal and pneumonia X-rays.

Another strength is that the project pipeline is organized into separate files for preprocessing, model construction, tuning, and evaluation. This makes the code easier to understand, run, and improve.

## 7.2 Limitations

The project has several limitations:

- The validation set is extremely small, making validation metrics unstable.
- The model over-predicts pneumonia and has low recall for the normal class.
- The uploaded code does not currently implement transfer learning, even though the README mentions it.

## 7.3 Challenges Encountered

One challenge was interpreting the model's performance. The training accuracy appeared very high, but the confusion matrix showed that the model still struggled with normal test images. This highlights why accuracy curves alone are not enough and why precision, recall, F1-score, confusion matrices, and ROC-AUC are necessary.

## 7.4 Future Improvements

Future improvements should include:

1. **Run additional algorithms:** Add Logistic Regression, SVM, Random Forest, and/or transfer learning models for comparison.
2. **Use data augmentation:** Apply random rotations, zooms, shifts, brightness changes, or flips to improve generalization.
3. **Improve class balancing:** Use stronger class weighting, oversampling, or balanced batch generation.
4. **Tune the decision threshold:** Adjust the classification threshold instead of using the default 0.5 threshold.
5. **Implement transfer learning:** Use pretrained models such as MobileNetV2, VGG16, ResNet50, or EfficientNet.

---

# 8. Conclusion

## 8.1 Summary of Results

This project developed a baseline CNN to classify chest X-ray images as normal or pneumonia. The CNN achieved approximately **76% test accuracy** and a **ROC-AUC of 0.93**. The model was especially strong at detecting pneumonia, with pneumonia recall of **0.99**, but it struggled to correctly identify normal cases, with normal recall of **0.37**.

The confusion matrix showed that the model correctly classified **388 out of 390 pneumonia images**, but it misclassified many normal images as pneumonia. This indicates that the model is highly sensitive to pneumonia but biased toward the pneumonia class.

## 8.2 Final Remarks

The results demonstrate that CNNs can learn useful visual features from chest X-ray images and can perform well at pneumonia detection. However, the model's imbalance between pneumonia detection and normal classification shows that further improvement is needed before the system could be considered reliable.

## 8.3 Future Work

Future work should focus on improving generalization, and reducing false positives.

---

# 9. Team Member Responsibilities

| Team Member | Responsibility |
|---|---|
| Jesus Villa | Project lead, project coordination, model selection strategy, baseline CNN implementation, and integration of project components. |
| Derek Shin | Data preprocessing, image resizing, normalization, dataset organization, and support for augmentation planning. |
| Momoka Aung | Exploratory data analysis, class distribution review, sample inspection, dataset visualization, and preprocessing validation. |
| Gauri Chahal | Model improvement and optimization, hyperparameter tuning planning, and implementation support for additional models or CNN variations. |
| Joshua Encinas | Performance analysis, confusion matrix and metric evaluation, visualization of results, documentation, and final report preparation. |

---

# References

1. Kaggle. **Chest X-Ray Images (Pneumonia)** dataset. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. GitHub repository. **chest-xray-pneumonia**. https://github.com/jvilla7/chest-xray-pneumonia
3. TensorFlow/Keras documentation. https://www.tensorflow.org/guide/keras
4. Scikit-learn documentation. https://scikit-learn.org/stable/
