Task: "Implement a model that predicts whether an incident will occur within the next H time steps based on the previous W steps of one or more time-series metrics. Use a sliding-window formulation and train the model using any standard machine-learning framework.

The applicant may use any suitable public dataset or generate a synthetic time series with labeled incident intervals (e.g. anomalies or threshold breaches). The emphasis is on correct problem formulation, model selection, training, and evaluation rather than dataset complexity or model size.

The solution should include a clear description of the modeling choices, the evaluation setup (including alert thresholds and metrics), and an analysis of the results. During follow-up, the applicant should be able to explain the design decisions, discuss limitations, and outline how the approach could be adapted to a real alerting system."

.

📈 Logistic Classification for Stock Spike Prediction
1. Problem Statement
The goal of this project is to predict whether a significant price movement (“spike”) will occur in the near future, using recent historical price data.

We formulate this as a binary classification problem:

Input: a rolling window of past returns

Output:

𝑦
=
{
1
if a spike occurs in the next 
𝐻
 days
0
otherwise
y={ 
1
0
​
  
if a spike occurs in the next H days
otherwise
​
 
A spike is defined as a price increase exceeding a chosen threshold (e.g. +3%) within a future horizon.

2. Methods and Modelling Choices
Model: Logistic Regression
We model the probability of a spike using logistic regression:

𝑃
(
𝑦
=
1
∣
𝑥
)
=
𝜎
(
𝑤
𝑇
𝑥
+
𝑏
)
P(y=1∣x)=σ(w 
T
 x+b)
where the sigmoid function is:

𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)= 
1+e 
−z
 
1
​
 
This provides:

A probabilistic interpretation

A simple and interpretable baseline

Fast and stable training

Feature Construction (Sliding Window)
We transform the time series into supervised data using a sliding window:

For each time step 
𝑘
k:

𝑥
𝑘
=
[
𝑟
𝑘
,
𝑟
𝑘
+
1
,
…
,
𝑟
𝑘
+
𝑊
−
1
]
x 
k
​
 =[r 
k
​
 ,r 
k+1
​
 ,…,r 
k+W−1
​
 ]
𝑦
𝑘
=
1
(
spike in 
𝑘
+
𝑊
 to 
𝑘
+
𝑊
+
𝐻
)
y 
k
​
 =1(spike in k+W to k+W+H)
This allows us to map temporal data into a standard classification framework.

Loss Function (Cross-Entropy)
The model is trained by minimising the binary cross-entropy loss:

𝐿
=
−
1
𝑁
∑
𝑖
=
1
𝑁
[
𝑦
𝑖
log
⁡
(
𝑝
𝑖
)
+
(
1
−
𝑦
𝑖
)
log
⁡
(
1
−
𝑝
𝑖
)
]
L=− 
N
1
​
  
i=1
∑
N
​
 [y 
i
​
 log(p 
i
​
 )+(1−y 
i
​
 )log(1−p 
i
​
 )]
Key Hyperparameters
Hyperparameter	Description
𝑊
W	Lookback window size
𝐻
H	Prediction horizon
Spike threshold	Defines positive class
𝐶
C	Regularisation strength
Class weights	Handle imbalance
Important Modelling Considerations
Class imbalance: spikes are rare → use class_weight="balanced"

Feature scaling: required for stable optimisation

Temporal dependence: overlapping windows introduce correlation

3. Implementation Overview
Data Pipeline
Retrieve price data (Yahoo Finance via yfinance)

Compute daily returns

Construct sliding window features

Generate labels based on future horizon

Split data using time-based split (no shuffling)

Model Training
Standardise features using StandardScaler

Train logistic regression on training set

Tune hyperparameters on validation set

4. Evaluation and Analysis
Metrics
Accuracy

Precision

Recall

F1-score

ROC-AUC

ROC Curve
We analyse model discrimination using the ROC curve:

True Positive Rate (TPR) vs False Positive Rate (FPR)

Helps evaluate performance across thresholds

Confusion Matrix
Provides insight into:

False positives (over-predicting spikes)

False negatives (missing spikes)

Important Insight
Due to overlapping windows:

Samples are not independent

Effective dataset size is smaller than it appears

Care must be taken to avoid data leakage

🔁 Improvements and Iterations
1. Baseline Logistic Classification
Used raw returns as features

Fixed classification threshold (0.5)

Default logistic regression

Limitations:

Poor recall for rare spike events

Sensitive to class imbalance

2. Validation-Tuned Logistic Classification
Improvements:

Introduced validation set

Tuned regularisation parameter 
𝐶
C

Adjusted classification threshold

Enabled class balancing

Outcome:

Improved recall

Better ROC-AUC

More robust performance

3. Logistic Classification with Extended Features
Extended feature set to improve representation:

Added Features
Rolling mean (trend)

Rolling volatility

Lagged returns

Momentum indicators

Motivation
Raw returns alone do not capture:

Market regimes

Volatility clustering

Trend persistence

Outcome
Improved class separability

Better predictive performance

Higher ROC-AUC

🚀 Future Improvements
Non-linear models (e.g. tree-based methods)

Sequential models (RNN / LSTM)

Reduced window overlap

Probability calibration

Strategy-level backtesting

⚠️ Key Takeaways
Logistic regression is a strong and interpretable baseline

Time-series classification requires careful handling of temporal structure

Feature engineering is critical for performance
