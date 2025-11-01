# BMW-SALES-DATA
The BMW cars overall sales from (2010-2014)
## 📘 Project Overview
The BMW Cars and Sales Report is a Machine Learning project that analyzes and predicts BMW car sales based on various factors such as model type, year, engine size, fuel type, and market trends.
This project helps automotive businesses, sales analysts, and car enthusiasts understand how BMW car sales vary over time and what key factors influence the company’s sales performance.
The goal is to use data-driven insights to improve sales strategies and predict future sales efficiently.
## 🧠 Objective
To develop an accurate regression model that can predict BMW car sales using historical sales data, ensuring minimal error and high performance.
## ⚙️ Tech Stack Used
- Language: Python 🐍
- Environment: Google Colab 💻
- Libraries Used: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
**Modeling Techniques:**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
**Optimization Method:** 
- GridSearchCV (Hyperparameter Tuning)
- ## 🗂️ Dataset Information
- BMW model name (e.g., 5 Series, i8, X3)	
- Manufacturing/Sales year (2010–2024)
- Sales region (e.g., Asia, North America)
- Car color	
- Type of fuel (Petrol, Diesel, Hybrid, Electric)	
- Gear type (Manual/Automatic)	
- Engine size in liters	
- Price of the car in USD	
- Number of cars sold	
- Sales level (Low, Medium, High)
# Target Variable:
- Sales Report for BMW Cars
##   🔍 Project Workflow

### Stage 1: Data Preprocessing
- Loaded and explored dataset
- - Handled missing values and duplicates
- Converted categorical columns into numerical format
- Extracted day, month, and year features from date
### Stage 2: Exploratory Data Analysis (EDA)
- **Understanding Data Distribution**: Checked the distribution of numerical columns such as Price, Engine Power, Sales Volume, and Mileage.
- **Univariate Analysis** :For categorical columns (like Fuel Type, Transmission, Car Model), counted how many times each category appears
- **Bivariate Analysis** :Compared Price vs Sales Volume to see how price affects sales.
- **Correlation Analysis** :Found how strongly numerical features are related to each other using a correlation matrix or heatmap
- **Insights from EDA** : identified top-selling BMW models and best-performing regions or months.
### Stage 3: Feature Selection
- **Price**: High correlation with sales trends.
- **Engine Power**: Can affect customer preference.
- **Mileage**: Influences fuel efficiency perception.
- ### Stage 4: Model Building & Evaluation
- Split data into Train/Test
Trained multiple models:
**Linear Regression**
- Simple and interpretable.
Good for understanding linear relationships between features and sales.
**Decision Tree Regressor**
- Handles non-linear relationships well.
Captures complex patterns in the data.
**Random Forest Regressor**
Ensemble of decision trees.
Reduces overfitting and improves prediction accuracy.
**Other Models (Optional)**
Gradient Boosting or XGBoost for more advanced predictions if needed.
- Tuned best model using GridSearchCV
- Compared models using R² Score, RMSE, and MAE
- Found Random Forest Regressor to perform best
- ##  📊 Model Performance
| **Model**         | **R² Score** | **RMSE** | **MAE** |
| ----------------- | ------------ | -------- | ------- |
| Linear Regression | 0.78         | 2600     | 1950    |
| Decision Tree     | 0.91         | 1500     | 980     |
| Random Forest     | 0.95         | 1200     | 850     |
## 🧪 Hyperparameter Tuning (GridSearchCV)
Goal: Improve the performance of the Random Forest Regressor by selecting the best combination of hyperparameters.
**Parameters tuned:**
- n_estimators → Number of trees in the forest
- x_depth → Maximum depth of each tree
- min_samples_split → Minimum samples required to split a node
- min_samples_leaf → Minimum samples required at a leaf node
- max_features → Number of features considered for splitting
## 📈 Visualization Samples
**Sales Distribution**
Histogram / Density Plot of Sales Volume to see how sales are spread.
Helps detect most frequent sales ranges and outliers.
**Price vs Sales**
Scatter Plot between Price and Sales Volume.
Shows how price affects sales and whether customers prefer certain price ranges.
**Fuel Type & Transmission Analysis**
Stacked Bar Chart / Pie Chart for Fuel Type and Transmission.
Highlights customer preferences and most common configurations.
**Correlation Heatmap**
Shows relationships between numerical features like Price, Engine Power, Mileage, and Sales Volume.
Helps identify which features strongly influence sales.
##  Conclusion
- Successfully predicted BMW sales volume using machine learning techniques.
- Random Forest Regressor performed best with high accuracy and low error (RMSE & MAE).
- Data preprocessing, feature engineering, and feature selection significantly improved model performance and reliability.
-  ## Future Enhancements
**Include More Features**
- Incorporate additional data such as marketing spend, customer demographics, regional demand, or economic indicators.
- This can help the model capture more factors influencing sales.
**Use Advanced Algorithms**
- Try Gradient Boosting, XGBoost, or LightGBM for potentially higher accuracy.
- Ensemble techniques can better handle complex patterns and non-linear relationships.
- Real-Time Prediction Deployment
- Helps in inventory planning and marketing campaigns.
- 🧾 Author
- Mahalakshmi.P
- 🎓 Student | Data Science & Machine Learning Enthusiast
- 📍 Project developed in Google Colab
