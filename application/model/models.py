import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import datetime
import datetime as dt
from itertools import product

#from sklearn.neural_network import MLPClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#from sklearn.linear_model import RidgeClassifier
#import lightgbm as lgb
#from xgboost import XGBClassifier
#from keras.layers import Dense, Dropout, LSTM, GRU, Embedding
#from keras.preprocessing.sequence import pad_sequences
#from keras.models import Sequential

class AI_Korea_Export():
    def __init__(self, raw_dataframe):

        self.model = SVC(gamma='auto')
        # DataFrame
        self.col_list = raw_dataframe.columns.tolist()
        self.raw = raw_dataframe

    def short_term_predict(self, lags = 12, test_split = 0.05):
        ans = dict()
        acc = list()
        
        for symbol in self.col_list:
            result = []
            # 1. Get one symbol data
            data = pd.DataFrame(self.raw[symbol])
            # 2. log returns
            data['returns'] = np.log(data / data.shift(1))
            data.dropna(inplace=True)
            # 3. Make lagged data
            cols = []
            for lag in range(1, lags+1):
                col = 'lag_{}'.format(lag)
                data[col] = data['returns'].shift(lag)
                cols.append(col)
            data.dropna(inplace=True)
            # 4. Make Binary data(One-hot encoding)
            data[cols] = np.where(data[cols] > 0, 1, 0)
            # 5. Make Direction label(Up / Down)
            data['direction'] = np.where(data['returns'] > 0, 1, -1)
            
            ## SVM, MLP Ensanble Models ##
            # 6. Split dataset(train:test = 95:5)
            split = int(len(data) * (1-test_split))
            train = data.iloc[:split].copy()
            test = data.iloc[split:].copy()

            # 8. Train
            self.model.fit(train[cols], train['direction'])

            train_acc = round(accuracy_score(train['direction'], self.model.predict(train[cols])), 4)
            # 9. Test
            test['position'] = self.model.predict(test[cols])
            test_acc = round(accuracy_score(test['direction'], test['position']), 4)
            
            # 10. Tommarow Prediction
            input_data = pd.DataFrame(np.array([test['direction'].tail(lags)]), columns=data[cols].columns)

            if self.model.predict(input_data).tolist()[0] == 1:
                result.append("Will Go Up ! ({}M)".format(str(dt.date.today().month+1)))
            
            else:
                result.append("Will DROP. ({}M)".format(str(dt.date.today().month+1)))

            ans[symbol.strip()] = result

        return ans # {symbol = [train_acc, test_acc, prediction]}

    def long_term_predict(self):
        ans = dict()
        sma1 = range(3, 17, 1)
        sma2 = range(24, 52, 2)

        for symbol in self.col_list:
            result = []
            # 1. Find Optimized MA-days
            results = pd.DataFrame()
            for SMA1, SMA2 in product(sma1, sma2):
                data = pd.DataFrame(self.raw[symbol])
                data.dropna(inplace=True)
                data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
                data['Short MA'] = data[symbol].rolling(SMA1).mean()
                data['Long MA'] = data[symbol].rolling(SMA2).mean()
                data.dropna(inplace=True)
                data['Position'] = np.where(data['Short MA'] > data['Long MA'], 1, -1)
                data['Strategy'] = data['Position'].shift(1) * data['Returns']
                data.dropna(inplace=True)
                perf = np.exp(data[['Returns', 'Strategy']].sum())
                results = results.append(pd.DataFrame(
                    {'Short MA': SMA1, 'Long MA': SMA2,
                    'MARKET': perf['Returns'],
                    'STRATEGY': perf['Strategy'],
                    'OUT': perf['Strategy'] - perf['Returns']},
                    index=[0]), ignore_index=True)
                
            data = pd.DataFrame(self.raw[symbol]).dropna()
            SMA1 = int(results.sort_values('OUT', ascending=False).iloc[0]['Short MA'])
            SMA2 = int(results.sort_values('OUT', ascending=False).iloc[0]['Long MA'])
            result.append(SMA1) # Short-MA
            result.append(SMA2) # Long-MA

            # 2. Calculate Short-MA
            data['Short MA'] = data[symbol].rolling(SMA1).mean()

            # 3. Calculate Long-MA
            data['Long MA'] = data[symbol].rolling(SMA2).mean()

            # 4. Calculate your Position
            data['Model Signal'] = np.where(data['Short MA'] > data['Long MA'], 1, -1)

            # 5. Save user position
            if data['Model Signal'].iloc[-1] == -1:
                result.append("Long-term 'SLUMP Signal'")
            else:
                result.append("Long-term 'BOOM Signal'")

            # 6. Plot
            #ax = data.plot(secondary_y='Model Signal', figsize=(12, 8))
            #ax.get_legend().set_bbox_to_anchor((0.215, 0.97))
            ans[symbol.strip()] = result

        return ans  # {symbol : [short-MA, long-MA, current position]}
