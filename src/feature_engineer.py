import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_wrds_data(crsp_df, funda_df=None, ibes_df=None):
    """
    处理从WRDS获取的原始数据，转换为统一的特征格式
    """
    # 确保日期格式正确
    crsp_df['date'] = pd.to_datetime(crsp_df['date'])
    
    # 创建以股票为列、日期为行的价格DataFrame
    price_pivot = crsp_df.pivot_table(
        index='date', 
        columns='ticker', 
        values='adj_prc'  # 使用调整后价格
    )
    
    # 计算基础特征
    features_dict = {}
    
    for ticker in price_pivot.columns:
        price_series = price_pivot[ticker].dropna()
        
        if len(price_series) < 100:  # 确保有足够数据
            continue
        
        # 计算5个核心特征（与之前相同，但基于WRDS数据）
        # 1. 动量
        momentum = price_series.pct_change(20)
        
        # 2. 波动率
        returns = price_series.pct_change()
        volatility = returns.rolling(window=20).std()
        
        # 3. 交易量变化（需要从crsp_df中获取交易量）
        if 'vol' in crsp_df.columns:
            vol_data = crsp_df[crsp_df['ticker'] == ticker].set_index('date')['vol']
            volume_change = vol_data.rolling(window=20).mean().pct_change(5)
        else:
            # 备用计算
            volume_change = price_series.rolling(window=20).mean().pct_change(5)
        
        # 4. RSI
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 5. 价格与移动平均线偏离度
        ma_60 = price_series.rolling(window=60).mean()
        price_ma_delta = (price_series - ma_60) / ma_60
        
        # 如果有基本面数据，添加基本面特征
        if funda_df is not None:
            # 这里简化处理，实际需要将季度数据对齐到日度
            pass
        
        features_dict[ticker] = pd.DataFrame({
            f'{ticker}_MOMENTUM': momentum,
            f'{ticker}_VOLATILITY': volatility,
            f'{ticker}_VOLUME_CHG': volume_change,
            f'{ticker}_RSI': rsi,
            f'{ticker}_PRICE_MA_DELTA': price_ma_delta
        })
    
    features_df = pd.concat(features_dict.values(), axis=1)
    features_df = features_df.dropna()
    
    print(f"特征数据形状: {features_df.shape}")
    return features_df, price_pivot

def create_labels_wrds(price_pivot, sp500_data, forward_periods=5):
    """
    使用WRDS数据创建标签
    """
    # 计算每只股票的日收益率
    stock_returns = price_pivot.pct_change()
    
    # 对齐标普500数据
    # 首先确保sp500_data有正确的日期索引
    if 'date' in sp500_data.columns:
        sp500_data.set_index('date', inplace=True)
    
    # 使用标普500总回报率作为基准
    if 'sp500_total_return' in sp500_data.columns:
        benchmark_returns = sp500_data['sp500_total_return'].pct_change()
    elif 'sp500_return' in sp500_data.columns:
        benchmark_returns = sp500_data['sp500_return']
    else:
        # 如果没有标普500数据，使用等权组合作为基准
        print("警告: 使用等权组合作为基准")
        benchmark_returns = stock_returns.mean(axis=1)
    
    # 确保基准收益率与股票收益率日期对齐
    aligned_returns = stock_returns.reindex(benchmark_returns.index).dropna()
    aligned_benchmark = benchmark_returns.reindex(aligned_returns.index)
    
    # 计算未来N日超额收益
    labels_dict = {}
    
    for ticker in aligned_returns.columns:
        # 计算股票未来N日累计收益
        stock_future_ret = (1 + aligned_returns[ticker]).rolling(window=forward_periods).apply(
            lambda x: x.prod(), raw=False
        ).shift(-forward_periods) - 1
        
        # 计算基准未来N日累计收益
        bench_future_ret = (1 + aligned_benchmark).rolling(window=forward_periods).apply(
            lambda x: x.prod(), raw=False
        ).shift(-forward_periods) - 1
        
        # 计算超额收益
        excess_return = stock_future_ret - bench_future_ret
        labels_dict[ticker] = excess_return
    
    labels_df = pd.DataFrame(labels_dict)
    labels_df.columns = [f'{col}_EXCESS_RETURN' for col in labels_df.columns]
    labels_df = labels_df.dropna()
    
    print(f"标签数据形状: {labels_df.shape}")
    return labels_df