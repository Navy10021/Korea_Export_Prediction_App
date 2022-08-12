# Korea Export Volume Prediction Using Artificial Intelligence models

This project aims to build a high-performance prediction model in the field of export by applying ML/DL models. We propose "AI-Korea export prediction", a light and fast export volume forecaster which is useful in real-life applications. This model predicts the direction of 'long-term' and 'short-term(Next month)' Korea export volume and can be used to determine the user decision making.

## 1. About App

 IMG
 

## 2. About Data


## 3. About models
We applied various 'Machine Learning', 'Ensemble' and 'Deep Learning' models to predict Korea export volume.

Machine Learing : Linear Regression(Ridge), Support Vector Machines
Ensemble : Random Forest, ExtraTrees, AdaBoost, XGBoost, Light GMB
Deep Learning : Multi-Layer Perceptron(MLP), Vanila RNN, LSTM, GRU
Transformer-based model(Doesn't apply to App) : Transformer

See "train_and_eval/Korea Export Prediction Results.ipnb" for performance evaluation of models.

## 4. About prediction

  - Short-term prediction : We use the AI models introduced above to predict the direction(up or down) of tomorrow's stock price.
  - Long-term prediction : After automatically finding the optimal moving average(MA) with our model, measures trend by the "Method of Moving Average technique". Then, predicts the user's stock position(sell or buy).
