
APP LINK :  https://decisiontree-cut2ymkd4awy7wcdapp8afx.streamlit.app/


# House Price Classification Using Decision Tree

This repository contains a complete workflow for predicting house price categories using a Decision Tree Classifier. The project demonstrates data preprocessing, exploratory data analysis, model training, evaluation, and saving the trained model for future use.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Model Training & Evaluation](#model-training--evaluation)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Results](#results)
- [Requirements](#requirements)
- [License](#license)
- [Author](#author)

---

## Project Overview

The goal of this project is to classify houses into different price categories based on features such as size, number of bedrooms, and other relevant attributes. The workflow includes:

- Data loading and preprocessing
- Exploratory data analysis (EDA)
- Feature selection and engineering
- Model training using Decision Tree Classifier
- Model evaluation using accuracy, confusion matrix, and regression metrics
- Feature importance analysis
- Saving the trained model for future predictions

---

## Dataset

The dataset used is `house_price_tree.csv`, which contains various features related to houses and their corresponding price categories.

**Sample columns:**
- `size_m2`: Size of the house in square meters
- `bedrooms`: Number of bedrooms
- `distance_to_city`: Distance to city center (km)
- `price_category`: Target variable (categorical)

You can replace or extend the dataset as needed for your use case.

---

## Installation

1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Set up a Python environment (recommended)**
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

---

## Usage

1. **Open the Jupyter Notebook**

   Open `House_price.ipynb` in Jupyter Notebook or Visual Studio Code.

2. **Run the notebook cells sequentially**

   The notebook covers:
   - Data loading
   - Data exploration and visualization
   - Model training and evaluation
   - Feature importance analysis
   - Sample prediction

3. **Sample Prediction**

   Example:
   ```python
   sample_input = [[120, 3, 7.5]]
   print("Prediction for (120, 3, 7.5):", dt.predict(sample_input)[0])
   ```

4. **Save the trained model**

   Uncomment the following lines in the notebook to save the trained model:
   ```python
   import pickle
   with open("house_price_model.pkl", "wb") as f:
       pickle.dump(dt, f)
   ```

---

## Code Structure

- `House_price.ipynb`: Main notebook containing all steps from data loading to model evaluation.
- `house_price_tree.csv`: Dataset file.
- `requirements.txt`: List of required Python packages.
- `house_price_model.pkl`: Saved model file (created after training).

---

## Model Training & Evaluation

- **Model:** Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`)
- **Metrics:**
  - Accuracy Score
  - Confusion Matrix
  - Feature Importances
  - Mean Squared Error
  - Mean Absolute Error
  - R2 Score

**Example output:**
```
Accuracy: 0.85
Confusion Matrix:
 [[30  5]
  [ 3 22]]
Feature Importances:
      Feature  Importance
0   size_m2        0.65
1   bedrooms       0.25
2   distance_to_city 0.10
```

---

## Saving and Loading the Model

To save the trained model:
```python
import pickle
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(dt, f)
```

To load the model for future predictions:
```python
import pickle
with open("house_price_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
prediction = loaded_model.predict([[120, 3, 7.5]])
```

---

## Results

- The model achieves good accuracy and provides insights into which features are most important for predicting house price categories.
- Visualizations help understand the distribution of price categories and relationships between features.

---

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

Install all dependencies using:
```sh
pip install -r requirements.txt
```

---

## License

This project is for educational purposes only.

---





