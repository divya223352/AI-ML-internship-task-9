# AI-ML-internship-task-9
Handling Imbalanced Data
In a typical fraud dataset, 99.9% of transactions are normal.
The Problem: If we use standard "Accuracy" as a metric, a model that predicts "Not Fraud" for every single case would be 99.9% accurate but would fail to catch any thieves.
The Solution: We use Stratified Sampling during the train-test split to ensure both sets have the same percentage of fraud. We also evaluate the model using Precision, Recall, and the F1-Score instead of just accuracy.
### 2. Why Random Forest?
Random Forest is an Ensemble Learning technique that uses "Bagging" (Bootstrap Aggregating).
How it works: It creates hundreds of individual Decision Trees on different subsets of the data.
Voting: Each tree gives a "vote" on whether a transaction is fraud or not. The final prediction is based on the majority vote.
Benefit: This reduces "overfitting" (where a model memorizes data instead of learning patterns) and handles the non-linear nature of credit card data better than a single Decision Tree.
### 3. Key Technical Terms
n_estimators: This is a hyperparameter that tells the model how many trees to build. We used 100, which is a standard balance between performance and speed.
Feature Importance: One of the best parts of Random Forest is that it tells us which variables (like transaction amount or location) were the biggest "red flags" for fraud.
StandardScaler: We use this to normalize the data. Since some features might be in dollars and others in timestamps, scaling them to a similar range helps models like Logistic Regression perform better.
Joblib: This is used for Model Persistence. It allows us to save the trained "brain" of our model as a .pkl file so we can use it later without retraining.
