# 🌱 Machine Learning Intro: Iris Dataset

---

## 📊 Dataset Source
- **UCI Machine Learning Repository**  
- [🔗 Iris Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

---

## 🧾 Dataset Overview
The Iris dataset is one of the most famous benchmark datasets in machine learning.  
It contains **150 samples** of iris flowers, each described by four numerical features:

- 🌸 **Sepal Length**  
- 🌸 **Sepal Width**  
- 🌸 **Petal Length**  
- 🌸 **Petal Width**

Each sample belongs to one of **three species**:
- *Iris-setosa*  
- *Iris-versicolor*  
- *Iris-virginica*

---

## ⚙️ Data Loading in Pandas
```python
import pandas as pd

# Load dataset directly from UCI repository
data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    header=None,
    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
)

# Preview first