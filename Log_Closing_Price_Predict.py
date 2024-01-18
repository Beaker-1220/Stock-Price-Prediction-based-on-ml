import pandas as pd
import torch
from autogluon.timeseries import TimeSeriesDataFrame,TimeSeriesPredictor



# 载入数据
data = pd.read_csv('processed_daily_data.csv', encoding='ANSI')
data['Date'] = pd.to_datetime(data['Date'])
data['item_id'] = 0

train_data = TimeSeriesDataFrame.from_data_frame(
    data,
    id_column="item_id",
    timestamp_column="Date",
)

predictor = TimeSeriesPredictor(
    prediction_length=48,
    path='autogluon-HSI',
    target='Log_Closing_Price',
    eval_metric="MASE",
    freq='1D'
)

predictor.fit(
    train_data,
    presets="best_quality",
    time_limit = 600
)

predictions = predictor.predict(train_data)
print(predictions)

# 确保predictions是一个DataFrame
if not isinstance(predictions, pd.DataFrame):
    predictions = pd.DataFrame(predictions)

# 保存predictions为CSV文件
predictions.to_csv('my_closing_price_predictions.csv', index=False)
