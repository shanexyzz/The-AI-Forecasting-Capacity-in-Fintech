import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import json
from local_preparation import calculate_technical_features

df = pd.read_parquet('./data/eikon_raw/analyst_data.parquet')
date_col = 'date'
if date_col in df.columns:
    # 强制把数字转换为日期对象
    # unit='ms' 代表毫秒，是金融数据最常用的单位。
    # 如果转换出来的年份不对（比如是 1970 年），试着改成 unit='s' (秒) 或 unit='ns' (纳秒)
    df[date_col] = pd.to_datetime(df[date_col], unit='ms')

#df.to_csv(f'./data/analyst_data.csv')
print(df.head())
