# Advanced-Machine-Learning-Coursework

---

## 1. Introduction
  The project is divided into two main parts: **regression** and **image analysis**.

---

## 3. Datasets and Project Submissions

Datasets for the project questions were provided by the instructors. All data is stored in **numpy** format.

---

## 4. Part 1 - Regression with Synthetic Data

### 4.1. First Problem - Basic Linear Regression

A simple linear regression problem with 100 training examples. The predictor is evaluated using the **sum of squared errors (SSE)** criterion.

### 4.2. Second Problem - Linear Regression with Outliers

This task is similar to the first but includes outliers in the training data. The challenge is to devise a strategy to estimate a predictor that can handle outliers effectively.

---

## 5. Part 2 - Image Analysis

The focus here is on analyzing butterfly images to classify two types of patterns in the wings: spots and eyespots. Pattern detection is carried out using the YOLO object detection method.

### 5.1. First Task

This is a binary classification task to predict the type of detected pattern. The input images are 30x30x3, considering RGB channels.

## 5.2 Second Task: Segmentation of Eyespots

The second task revolves around the segmentation of a specific type of eyespot found in butterflies of the *bicyclus anynana* species. These eyespots display a distinct white center, bordered by a black ring, which is then encompassed by a golden ring. The primary objective is to segment these into three separate areas:
1. The white center
2. The combined black and golden ring
3. The background

This challenge is essentially a pixel classification task. The aim is to categorize each pixel within a 30x30 RGB eyespot image, leveraging a 5x5 neighborhood of pixels surrounding it as features. The classification process is a multiclass one, with labels ranging from 0 to 2, representing the background, rings, and white center, respectively. The segmentation masks utilized for training were manually crafted by highlighting the three aforementioned areas and labeling all pixels within a single region consistently.

It's essential to note that the dataset for this task exhibits an imbalance. There's a significant disparity in the pixel count across the three eyespot regions.


