Task: 
"Implement a model that predicts whether an incident will occur within the next H time steps based on the previous W steps of one or more time-series metrics. Use a sliding-window formulation and train the model using any standard machine-learning framework.

The applicant may use any suitable public dataset or generate a synthetic time series with labeled incident intervals (e.g. anomalies or threshold breaches). The emphasis is on correct problem formulation, model selection, training, and evaluation rather than dataset complexity or model size.

The solution should include a clear description of the modeling choices, the evaluation setup (including alert thresholds and metrics), and an analysis of the results. During follow-up, the applicant should be able to explain the design decisions, discuss limitations, and outline how the approach could be adapted to a real alerting system."


## Logistic Classification for Time-Series Incident Prediction

TLDR: Given a window of recent price returns, can we predict whether a spike is coming in the next few days? This project frames that as a binary classification problem and solves it with logistic regression. I iterate through three versions: a simple baseline, one with proper validation and hyperparameter tuning, and a final version with added features (rolling mean, volatility, momentum). Each version improves on the last. The framework is intentionally general, and you can easily swap out price data for CPU usage or API latency and the same approach applies.

## 1. Problem Statement

The goal of this project is to predict whether an incident will occur within the next H time steps, based on the previous W steps of a time series.

We can formulate this as a **binary classification problem**:

- Input: a rolling window of past returns (previous \( W \) steps)  
- Output:

$$
y =
\begin{cases}
1 & \text{if an incident occurs in the next } H \text{ steps} \\
0 & \text{otherwise}
\end{cases}
$$

In this implementation, I have chosen my time-series data to be the financial data (SPY ETF) from Yahoo Finance, and I have defined the incident to be a "spike", which is:

> A price increase exceeding a threshold within the next H days.

---

## 2. Methods and Modelling Choices

### Model: Logistic Regression

I have chosen to model the probability of an incident using logistic regression:

$$
P(y=1 \mid x) = \sigma(w^T x + b)
$$

where:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$


---

### Sliding Window (Feature/Inputs) Formulation

We convert the time series into supervised data:

$$
x_k = [r_k, r_{k+1}, \dots, r_{k+W-1}]
$$

$$
y_k = \mathbf{1}(\text{incident in } k+W \text{ to } k+W+H)
$$


---

### Loss Function

We minimise binary cross-entropy:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[
y_i \log(p_i) + (1 - y_i)\log(1 - p_i)
\right]
$$

---

### Key Hyperparameters

The Hyperparameters for this model are:

| Hyperparameter | Description |
|------|--------|
|  W  | Lookback window size |
|  H  | Prediction horizon |
|  theta  | Incident threshold |
|  C  | Regularisation strength |
|  tau  | Decision threshold |

---

## 3. Overview of Model Implementation

### Building Datasets

1. Downloaded data using `yfinance`  
2. Computed daily returns  
3. Built sliding window dataset  
4. Generated true labels using future horizon  
5. Split dataset into training/validation/test data sets (60/20/20)

---

### Training of Model

- Standardised features (better convergence)
- Trained logistic regression  
- Tuned hyperparameters using validation set  

---

### Evaluating Model Results

I have evaluated the results of my models using a few metrics, including:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

---

### Threshold ( tau )

The model ultimately outputed probabilities, which were then converted to binary classifications (incident or no incident) according to a threshold:

$$
\hat{y} =
\begin{cases}
1 & \text{if } P(y=1 \mid x) \geq \tau \\
0 & \text{otherwise}
\end{cases}
$$
 

---

### Additional Analysis

- ROC Curve → discrimination ability  
- Confusion Matrix → error breakdown  

---

## 4. Model Development and Iterations

This project was developed in three stages:

---

### Baseline Logistic Classification  
"Simple Logistic Classification.py"

**Setup:**
- Fixed hyperparameters:
  - W = 20, H = 5, theta = 3%
- Features: raw past returns only  
- Fixed threshold tau = 0.5

**Observations:**
- Simple and interpretable  
- Poor recall for rare events  
- Sensitive to class imbalance  

**Key Limitation:**
- No hyperparameter tuning  
- Limited feature representation  

---

### Validation-Tuned Logistic Classification  
"Validation Logistic Classification.py"

**Improvements:**
- Introduced validation set  
- Performed grid search over:
  - W, H, theta, C
- Tuned decision threshold tau
- Selected model using validation ROC-AUC

**Observations:**
- Significant improvement in generalisation  
- Better separation between classes  
- More stable performance  

**Insight:**
- Model performance is highly sensitive to:
  - Window size  
  - Prediction horizon  
  - threshold definition  

---

### Validation + Feature Engineering  
"Validation LC with more features.py"

**Additional Features:**
- Rolling mean: 5-day and 10-day means  
- Rolling volatility: 5-day and 10-day volatilities
- Momentum: 20-day return  

**Additional Improvement:**
- Used `class_weight="balanced"` to handle imbalance  

**Observations:**
- Improved class separation  
- Higher ROC-AUC  
- More informative features  


---

## 5. Limitations

- Assumed samples are independent (unlikely due to overlap)  
- Linear model → cannot capture complex dynamics  
- Labels depend on arbitrary threshold (theta)

---

## 6. Future Improvements and Potential Methods

- Tree-based models (XGBoost, Random Forest)  
- Sequential models (LSTM, Transformer)  
- Reduce window overlap  
- Calibrate probabilities  

