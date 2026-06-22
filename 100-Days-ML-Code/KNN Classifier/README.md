## Functions and Methods Used in Detail

This notebook demonstrates a complete machine learning workflow, from data loading and preprocessing to model training, evaluation, and interpretation. Below is a detailed explanation of the key functions and methods used at each stage.

### 1. Data Loading and Initial Inspection

*   **`pandas.read_csv(path, header=None)`**:
    *   **Purpose**: Reads a comma-separated values (CSV) file into a DataFrame.
    *   **Usage in notebook**: `dataset = pd.read_csv(path, header=None)` was used to load the breast-cancer-wisconsin.txt file. `header=None` indicates that the dataset does not have a header row, so pandas automatically assigns integer column names.

*   **`dataset.iloc[:, [start_col, end_col]].values`**:
    *   **Purpose**: Integer-location based indexing for selection by position. It's used to select specific rows and columns by their integer indices.
    *   **Usage in notebook**: `X = dataset.iloc[:, [2, 3]].values` and `Y = dataset.iloc[:, 4].values` were initially used to extract features (columns 2 and 3) and the target variable (column 4) from the raw dataset. This was later updated to select all features for `X` and the `Class` column for `y` after column renaming and dropping the 'Id' column.

*   **`dataset.shape`**:
    *   **Purpose**: Returns a tuple representing the dimensionality of the DataFrame (rows, columns).
    *   **Usage in notebook**: `dataset.shape` was used to get a quick overview of the dataset's size.

*   **`dataset.head()`**:
    *   **Purpose**: Returns the first `n` rows of the DataFrame. Useful for quickly checking data structure and content.
    *   **Usage in notebook**: `dataset.head()` was used multiple times to inspect the initial rows of the DataFrame before and after column renaming.

### 2. Data Preprocessing and Cleaning

*   **`dataset.columns = col_names`**:
    *   **Purpose**: Assigns a new list of column names to the DataFrame.
    *   **Usage in notebook**: A list `col_names` was created, and then `dataset.columns = col_names` was used to give meaningful names to the columns, improving readability and understanding of the dataset.

*   **`dataset.drop('Id', axis=1, inplace=True)`**:
    *   **Purpose**: Removes specified rows or columns from a DataFrame.
    *   **Usage in notebook**: `dataset.drop('Id', axis=1, inplace=True)` was used to remove the 'Id' column, which was identified as redundant and not contributing to the predictive power of the model. `axis=1` specifies dropping a column, and `inplace=True` modifies the DataFrame directly.

*   **`dataset.info()`**:
    *   **Purpose**: Prints a concise summary of a DataFrame, including index dtype, column dtypes, non-null values, and memory usage.
    *   **Usage in notebook**: `dataset.info()` was used to check data types and identify non-null counts, which helped in detecting that 'Bare_Nuclei' was an object type despite containing numerical data, indicating potential issues.

*   **`dataset[var].value_counts()`**:
    *   **Purpose**: Returns a Series containing counts of unique values for a given column.
    *   **Usage in notebook**: Used in a loop (`for var in dataset.columns: print(dataset[var].value_counts())`) to examine the frequency distribution of values for each feature, helping to identify inconsistent entries (e.g., '?' in 'Bare_Nuclei').

*   **`pandas.to_numeric(dataset['Bare_Nuclei'], errors='coerce')`**:
    *   **Purpose**: Converts an argument to a numeric type. `errors='coerce'` will turn unparseable values into `NaN` (Not a Number).
    *   **Usage in notebook**: `dataset['Bare_Nuclei'] = pd.to_numeric(dataset['Bare_Nuclei'], errors='coerce')` was crucial for converting the 'Bare_Nuclei' column to a numeric type, correctly handling the '?' characters by converting them to `NaN`.

*   **`dataset.dtypes`**:
    *   **Purpose**: Returns a Series with the data type of each column.
    *   **Usage in notebook**: `dataset.dtypes` was used after `pd.to_numeric` to confirm that 'Bare_Nuclei' was successfully converted to a numeric (float64) type.

*   **`dataset.isnull().sum()`** / **`dataset.isna().sum()`**:
    *   **Purpose**: Returns the number of missing (NaN) values in each column. Both methods achieve the same result.
    *   **Usage in notebook**: These were used to identify columns with missing values and specifically to confirm the presence of 16 missing values in 'Bare_Nuclei' after coercion.

*   **`dataset['Bare_Nuclei'].unique()`**:
    *   **Purpose**: Returns unique values in a Series. Useful for inspecting the distinct entries in a column.
    *   **Usage in notebook**: `dataset['Bare_Nuclei'].unique()` was used to see all distinct values, including the `NaN` introduced, confirming the '?' values were handled.

*   **`X_train[col].median()`**:
    *   **Purpose**: Calculates the median value of a Series (column).
    *   **Usage in notebook**: `col_median = X_train[col].median()` was used to compute the median of each column in the training set, specifically for 'Bare_Nuclei', to be used for imputation.

*   **`df1[col].fillna(col_median)`**:
    *   **Purpose**: Fills `NaN` values using a specified method or value.
    *   **Usage in notebook**: `df1[col].fillna(col_median)` was used to impute missing values in both `X_train` and `X_test` with the median calculated *from the training set*. This is a standard practice to prevent data leakage from the test set.

### 3. Exploratory Data Analysis (EDA) and Visualization

*   **`dataset.describe()`**:
    *   **Purpose**: Generates descriptive statistics of DataFrame columns, including count, mean, standard deviation, min, max, and quartiles.
    *   **Usage in notebook**: `print(round(dataset.describe(),2))` provided a statistical summary of the numerical variables, showing central tendency, dispersion, and range.

*   **`matplotlib.pyplot.rcParams['figure.figsize'] = (width, height)`**:
    *   **Purpose**: Customizes Matplotlib's default settings, in this case, the default figure size.
    *   **Usage in notebook**: `plt.rcParams['figure.figsize']=(30,25)` was used to set a large figure size for plotting multiple histograms in a single view.

*   **`dataset.plot(kind='hist', bins=10, subplots=True, layout=(rows, cols), sharex=False, sharey=False)`**:
    *   **Purpose**: Generates various types of plots from DataFrame columns. `kind='hist'` creates histograms.
    *   **Usage in notebook**: `dataset.plot(kind='hist', bins=10, subplots=True, layout=(5,2), sharex=False, sharey=False)` was used to visualize the distribution of all numerical variables, checking for normality or skewness.

*   **`matplotlib.pyplot.show()`**:
    *   **Purpose**: Displays all open Matplotlib figures.
    *   **Usage in notebook**: `plt.show()` was called after plotting to render the visualizations.

*   **`dataset.corr()`**:
    *   **Purpose**: Computes pairwise correlation of columns, excluding NA/null values.
    *   **Usage in notebook**: `correlation = dataset.corr()` calculated the correlation matrix between all features and the target variable.

*   **`correlation['Class'].sort_values(ascending=False)`**:
    *   **Purpose**: Selects the 'Class' column from the correlation matrix and sorts its values.
    *   **Usage in notebook**: Used to identify features that are most highly correlated (positively or negatively) with the target variable 'Class'.

*   **`matplotlib.pyplot.figure(figsize=(width, height))`**:
    *   **Purpose**: Creates a new figure, or activates an existing figure, with a specified size.
    *   **Usage in notebook**: `plt.figure(figsize=(10,8))` was used to define the size of the heatmap plot.

*   **`seaborn.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')`**:
    *   **Purpose**: Plots rectangular data as a color-encoded matrix. A common way to visualize correlation matrices.
    *   **Usage in notebook**: `sns.heatmap(...)` was used to create a correlation heatmap, visually representing the relationships between all variables. `annot=True` shows the correlation values on the map, and `fmt='.2f'` formats them to two decimal places.

*   **`a.set_xticklabels(a.get_xticklabels(), rotation=90)`** / **`a.set_yticklabels(a.get_yticklabels(), rotation=30)`**:
    *   **Purpose**: Sets the labels for the x and y axes of a plot, allowing for rotation to prevent overlap.
    *   **Usage in notebook**: These methods were used to rotate the tick labels on the heatmap for better readability, especially when dealing with many column names.

### 4. Model Training and Evaluation

*   **`sklearn.model_selection.train_test_split(X, y, test_size=float, random_state=int)`**:
    *   **Purpose**: Splits arrays or matrices into random train and test subsets.
    *   **Usage in notebook**: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)` divided the dataset into training (80%) and testing (20%) sets. `random_state` ensures reproducibility of the split.

*   **`sklearn.preprocessing.StandardScaler()`**:
    *   **Purpose**: Standardize features by removing the mean and scaling to unit variance. It transforms data such that its distribution has a mean of 0 and a standard deviation of 1.
    *   **Usage in notebook**: An instance of `StandardScaler` was created (`scaler = StandardScaler()`) to scale the feature data.

*   **`scaler.fit_transform(X_train)`**:
    *   **Purpose**: Fits the scaler to the training data and then transforms it. `fit()` computes the mean and standard deviation, and `transform()` applies this scaling.
    *   **Usage in notebook**: `X_train = scaler.fit_transform(X_train)` was used to scale the training features. It's crucial to fit *only* on the training data to avoid data leakage.

*   **`scaler.transform(X_test)`**:
    *   **Purpose**: Transforms the test data using the parameters (mean and standard deviation) learned from the training data during the `fit` step.
    *   **Usage in notebook**: `X_test = scaler.transform(X_test)` was used to scale the test features using the `scaler` fitted on `X_train`.

*   **`pandas.DataFrame(data, columns=[cols])`**:
    *   **Purpose**: Creates a DataFrame from data. Here, it's used to convert NumPy arrays back into DataFrames while preserving column names.
    *   **Usage in notebook**: `X_train = pd.DataFrame(X_train,columns=[cols])` and `X_test=pd.DataFrame(X_test,columns=[cols])` were used to convert the scaled NumPy arrays back into pandas DataFrames with their original column names, which is helpful for inspection and consistency.

*   **`sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)`**:
    *   **Purpose**: Implements the K-Nearest Neighbors (KNN) classification algorithm, which classifies a data point based on the majority class of its `k` nearest neighbors.
    *   **Usage in notebook**: `knn = KNeighborsClassifier(n_neighbors=3)` (and for k=5, 6, 7, 8, 9) created instances of the KNN classifier with varying numbers of neighbors.

*   **`knn.fit(X_train, y_train)`**:
    *   **Purpose**: Trains the KNN classifier model using the training data.
    *   **Usage in notebook**: `knn.fit(X_train, y_train)` was called to train the model on the scaled training features and target labels.

*   **`knn.predict(X_test)`**:
    *   **Purpose**: Predicts class labels for samples in `X_test` using the trained classifier.
    *   **Usage in notebook**: `y_pred = knn.predict(X_test)` generated predictions on the test set, and `y_pred_train = knn.predict(X_train)` for the training set.

*   **`knn.predict_proba(X_test)`**:
    *   **Purpose**: Returns probability estimates for each class for the samples in `X_test`.
    *   **Usage in notebook**: `knn.predict_proba(X_test)` was used to get the probabilities of a sample belonging to 'benign cancer' (class 2) or 'malignant cancer' (class 4). `[:,0]` for class 2 probabilities and `[:,1]` for class 4 probabilities.

*   **`sklearn.metrics.accuracy_score(y_true, y_pred)`**:
    *   **Purpose**: Computes subset accuracy classification score. The set of labels predicted for a sample must exactly match the corresponding set of labels in `y_true`.
    *   **Usage in notebook**: `accuracy_score(y_test, y_pred)` was used extensively to evaluate the model's performance on both training and test sets.

*   **`knn.score(X, y)`**:
    *   **Purpose**: Returns the mean accuracy on the given test data and labels. It's a convenience method provided by scikit-learn estimators.
    *   **Usage in notebook**: `knn.score(X_train, y_train)` and `knn.score(X_test, y_test)` were used to get the training and test set scores directly from the model object.

*   **`y_test.value_counts()`**:
    *   **Purpose**: Returns a Series containing counts of unique values in `y_test`. Used to understand the class distribution.
    *   **Usage in notebook**: Used to determine the most frequent class in the test set to calculate the null accuracy score.

*   **`sklearn.metrics.confusion_matrix(y_true, y_pred)`**:
    *   **Purpose**: Computes a confusion matrix to evaluate the accuracy of a classification. The output matrix `C` is such that `C_{ij}` is equal to the number of observations known to be in group `i` and predicted to be in group `j`.
    *   **Usage in notebook**: `cm = confusion_matrix(y_test, y_pred)` calculated the confusion matrix, which was then used to derive True Positives, True Negatives, False Positives, and False Negatives.

*   **`sklearn.metrics.classification_report(y_true, y_pred)`**:
    *   **Purpose**: Builds a text report showing the main classification metrics: precision, recall, f1-score, and support for each class.
    *   **Usage in notebook**: `print(classification_report(y_test, y_pred_7))` provided a comprehensive summary of the classification performance for the kNN model with k=7.

*   **`sklearn.metrics.roc_curve(y_true, y_score, pos_label=label)`**:
    *   **Purpose**: Computes Receiver Operating Characteristic (ROC) curve. It computes the True Positive Rate (TPR), False Positive Rate (FPR), and thresholds for plotting the ROC curve.
    *   **Usage in notebook**: `fpr, tpr, thresholds = roc_curve(y_test, y_pred_1, pos_label=4)` was used to obtain the necessary data points for plotting the ROC curve, with `pos_label=4` indicating 'malignant cancer' as the positive class.

*   **`sklearn.metrics.roc_auc_score(y_true, y_score)`**:
    *   **Purpose**: Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    *   **Usage in notebook**: `ROC_AUC = roc_auc_score(y_test, y_pred_1)` calculated the ROC AUC score, providing a single metric to summarize the classifier's performance across all possible classification thresholds.

*   **`sklearn.model_selection.cross_val_score(estimator, X, y, cv=int, scoring='metric')`**:
    *   **Purpose**: Evaluates a score by cross-validation. It performs k-fold cross-validation and returns an array of scores, one for each fold.
    *   **Usage in notebook**: `Cross_validated_ROC_AUC = cross_val_score(knn_7, X_train, y_train, cv=5, scoring='roc_auc').mean()` calculated the cross-validated ROC AUC, providing a more robust estimate of model performance. Similarly, `scores = cross_val_score(knn_7, X_train, y_train, cv = 10, scoring='accuracy')` was used for k-fold cross-validation with accuracy as the scoring metric.

*   **`scores.mean()`**:
    *   **Purpose**: Calculates the average of the scores obtained from cross-validation.
    *   **Usage in notebook**: `print('Average cross-validation score: {:.4f}'.format(scores.mean()))` reported the average accuracy across all folds.