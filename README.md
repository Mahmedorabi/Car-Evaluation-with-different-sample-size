# Car Evaluation with Different Sample Size

This project evaluates the performance of a K-Nearest Neighbors (KNN) classifier on the Car Evaluation dataset with varying sample sizes and K values. The goal is to understand how the size of the training data and the choice of K value affect the model's performance in terms of accuracy and training time.

## Project Overview

The notebook `Car Evaluation with different sample size.ipynb` explores the following:
- Loading and preprocessing the Car Evaluation dataset.
- Splitting the data into training and testing sets.
- Training a KNN classifier with different portions of the training data and different K values.
- Evaluating the model's performance in terms of accuracy and training time.
- Visualizing the results.

## Setup Instructions


1. **Download the dataset:**
    The Car Evaluation dataset is available from the UCI Machine Learning Repository. Download it and place it in the appropriate directory.

## Notebook Structure

The Jupyter Notebook `Car Evaluation with different sample size.ipynb` contains the following sections:

1. **Importing Libraries:**
    Necessary libraries such as pandas, numpy, matplotlib, seaborn, scikit-learn, and time are imported.

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import time
    ```

2. **Loading and Preprocessing Data:**
    The Car Evaluation dataset is loaded, and preprocessing steps such as encoding categorical variables are applied.

    ```python
    # Load data and show it
   df=pd.read_csv("Data set/CAR_EVALUATION.csv")
    # Encoding categorical data
    label=LabelEncoder()
    for col in df.columns:
    df[col]=label.fit_transform(df[col])
    ```
3. **Visualizing Data**
   Categorical data distribution with target
   
   ![categorical data car](https://github.com/Mahmedorabi/Car-Evaluation-with-different-sample-size/assets/105740465/68fe6bb2-ab37-487a-814e-c89f4825ddd7)

4. **Splitting Data:**
    The data is split into training,testing and validation sets.

    ```python

   # Splitting the dataset into Features and Target variables.
   x=df.drop("Target",axis=1)
   y=df['Target']
    # spilt data into train,test and validation
   x_temp,x_test,y_temp,y_test=train_test_split(x,y,test_size=428,random_state=42,shuffle=True)
   x_train,x_val,y_train,y_val=train_test_split(x_temp,y_temp,test_size=300,random_state=42,shuffle=True)

    ```

5. **Evaluating the Model:**
    The KNN classifier is trained with different portions of the training data and different K values. The accuracy and training time are measured.
    ##### Try to use different number of training samples to show the impact of number of training samples. Use 10%,20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% and 100% of the training set for 10 separate KNN classifiers and show their performance (accuracy score) on the validation set and testing set. You can specify a fixed K=2 value (nearest neighbor) in this question. Notably, X axis is the portion of the training set, Y axis should be the accuracy score. There should be two lines in total, one is for the validation set and another is for the testing set
     ![ACC of car](https://github.com/Mahmedorabi/Car-Evaluation-with-different-sample-size/assets/105740465/9b7b827c-ed81-4e30-986f-cd9a641c66e0)
     ##### Use 100% of training samples, try to find the best K value, and show the accuracy curve on the validation set when K varies from 1 to 10.
      ![ACC k of car](https://github.com/Mahmedorabi/Car-Evaluation-with-different-sample-size/assets/105740465/f6c770a0-560c-4081-8735-f2b3accf4a17)

6. **Visualizing Results:**
    The results are visualized using bar plots for accuracy and training time.

   
    ![Car time](https://github.com/Mahmedorabi/Car-Evaluation-with-different-sample-size/assets/105740465/17ca0267-ad06-4a78-b714-907efc503068)
    

## Results

The project demonstrates how different sample sizes and K values impact the KNN classifier's performance. The visualizations provide insights into the trade-offs between accuracy and training time.

## Conclusion

This project helps understand the effects of sample size and K value on the performance of a KNN classifier, providing a practical guide for choosing the right parameters for optimal performance.

