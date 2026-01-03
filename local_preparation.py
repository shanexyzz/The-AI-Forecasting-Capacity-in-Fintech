# local_preparation.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import json

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入配置
from src.config import Config

# 1. 导入我们之前写好的模块
from src.wrds_fetcher import WRDSFetcher
from src.eikon_fetcher import EikonFetcher

def prepare_and_save_datasets():
    """本地数据准备主函数"""
    print("="*60)
    print("阶段一：本地数据准备 (整合WRDS + Eikon)")
    print("="*60)
    
    # 创建所需目录
    Config.setup_directories()
    
    # A. 从WRDS获取核心数据 (价格 + 基本面)
    print("\n[A] 从WRDS获取价格与基本面数据...")
    
    # 获取WRDS用户名（如果未在配置中设置）
    wrds_username = Config.WRDS_USERNAME
    if wrds_username is None:
        wrds_username = input("请输入WRDS用户名: ")
    
    wrds_fetcher = WRDSFetcher(wrds_username=wrds_username)
    
    # 获取CRSP调整后价格
    crsp_data = wrds_fetcher.fetch_crsp_data(
        Config.TECH_TICKERS, 
        Config.START_DATE, 
        Config.END_DATE
    )
    
    # 获取标普500指数作为基准
    sp500_data = wrds_fetcher.fetch_sp500_index(Config.START_DATE, Config.END_DATE)
    
    # B. 从本地Eikon获取IBES分析师数据
    print("\n[B] 从本地Eikon获取IBES分析师预期数据...")
    
    # 获取Eikon App Key（如果未在配置中设置）
    eikon_app_key = Config.EIKON_APP_KEY
    if eikon_app_key is None:
        eikon_app_key = input("请输入Eikon App Key: ")
    
    eikon_fetcher = EikonFetcher(app_key=eikon_app_key)
    
    # 使用Eikon获取分析师预测
    analyst_data = eikon_fetcher.fetch_analyst_estimates(Config.TECH_TICKERS)
    
    wrds_fetcher.close()
    
    # C. 特征工程 (整合WRDS价格 + Eikon分析师数据)
    print("\n[C] 特征工程 - 整合所有数据源...")
    
    # C.1 处理WRDS价格数据生成基础特征
    price_pivot = crsp_data.pivot_table(
        index='date', 
        columns='ticker', 
        values='adj_prc'
    )
    
    # 计算技术指标特征
    base_features = calculate_technical_features(price_pivot)
    
    # C.2 整合分析师数据特征
    if analyst_data is not None and not analyst_data.empty:
        analyst_features = process_analyst_features(analyst_data, price_pivot.index, price_pivot)
        # 合并基础特征和分析师特征
        all_features = pd.concat([base_features, analyst_features], axis=1)
        print(f"已整合分析师特征: {len(analyst_features.columns)}个特征")
    else:
        all_features = base_features
        print("警告: 未获取到分析师数据，仅使用技术特征")   
    
    # D. 创建标签 (未来超额收益)
    print("\n[D] 创建预测标签...")
    labels = create_labels_from_wrds(price_pivot, sp500_data, Config.PREDICTION_HORIZON)
    
    # E. 确保特征和标签对齐
    print("\n[E] 对齐特征与标签...")
    aligned_features, aligned_labels = align_features_labels(all_features, labels)
    
    # 打印数据统计
    print(f"\n数据统计:")
    print(f"- 特征维度: {aligned_features.shape[1]}个特征")
    print(f"- 样本数量: {aligned_features.shape[0]}个交易日")
    print(f"- 股票数量: {len(Config.TECH_TICKERS)}只股票")
    print(f"- 数据时间范围: {aligned_features.index.min().date()} 到 {aligned_features.index.max().date()}")
    
    # F. 保存为标准化格式
    print("\n[F] 保存标准化数据文件...")
    
    # 创建数据目录
    output_dir = './data_for_colab'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存特征和标签
    aligned_features.to_parquet(f'{output_dir}/features.parquet')
    aligned_labels.to_parquet(f'{output_dir}/labels.parquet')
    
    # 保存价格数据用于回测
    price_pivot.to_parquet(f'{output_dir}/price_data.parquet')
    
    # 保存数据信息元数据
    meta_info = {
        'tickers': Config.TECH_TICKERS,
        'benchmark': Config.BENCHMARK_TICKER,
        'feature_columns': list(aligned_features.columns),
        'label_columns': list(aligned_labels.columns),
        'date_range': {
            'start': aligned_features.index.min().strftime('%Y-%m-%d'),
            'end': aligned_features.index.max().strftime('%Y-%m-%d')
        },
        'num_samples': len(aligned_features),
        'prediction_horizon': Config.PREDICTION_HORIZON,
        'sequence_length': Config.SEQUENCE_LENGTH,
        'preparation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    import json
    with open(f'{output_dir}/data_info.json', 'w') as f:
        json.dump(meta_info, f, indent=2)
    
    print("\n" + "="*60)
    print("本地数据准备完成！")
    print(f"特征文件: {output_dir}/features.parquet ({len(aligned_features.columns)}个特征)")
    print(f"标签文件: {output_dir}/labels.parquet ({len(aligned_labels.columns)}只股票)")
    print(f"价格文件: {output_dir}/price_data.parquet (用于回测)")
    print(f"配置文件: {output_dir}/data_info.json (数据信息)")
    print("\n请将 '{output_dir}' 文件夹上传到Google Drive或Colab环境。")
    print("="*60)
    
    return aligned_features, aligned_labels, price_pivot

def calculate_technical_features(price_pivot):
    """计算技术指标特征"""
    features_dict = {}
    
    for ticker in price_pivot.columns:
        price = price_pivot[ticker].dropna()
        
        if len(price) < 100:  # 确保有足够数据
            continue
        
        # 1. 动量 (20日收益率)
        momentum = price.pct_change(20)
        
        # 2. 波动率 (20日收益率标准差)
        returns = price.pct_change()
        volatility = returns.rolling(20).std()
        
        # 3. RSI (14日)
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 4. 价格偏离度 (价格与60日均线的偏离百分比)
        ma_60 = price.rolling(60).mean()
        price_ma_delta = (price - ma_60) / ma_60
        
        # 5. 短期/长期均线比率 (10日/50日)
        ma_short = price.rolling(10).mean()
        ma_long = price.rolling(50).mean()
        ma_ratio = ma_short / ma_long - 1
        
        # 6. 布林带位置
        bb_ma = price.rolling(20).mean()
        bb_std = price.rolling(20).std()
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        bb_position = (price - bb_lower) / (bb_upper - bb_lower)
        
        # 7. 价格通道突破 (20日最高/最低)
        high_20 = price.rolling(20).max()
        low_20 = price.rolling(20).min()
        channel_breakout = (price - high_20.shift(1)) / price  # 突破上轨
        channel_breakdown = (price - low_20.shift(1)) / price  # 突破下轨
        
        # 8. 价格加速度 (动量的动量)
        momentum_acceleration = momentum.diff(5)
        
        features_dict[ticker] = pd.DataFrame({
            f'{ticker}_MOM_20D': momentum,
            f'{ticker}_VOL_20D': volatility,
            f'{ticker}_RSI_14D': rsi,
            f'{ticker}_MA_DELTA_60D': price_ma_delta,
            f'{ticker}_MA_RATIO_10_50D': ma_ratio,
            f'{ticker}_BB_POSITION': bb_position,
            f'{ticker}_CHANNEL_BREAKOUT': channel_breakout,
            f'{ticker}_CHANNEL_BREAKDOWN': channel_breakdown,
            f'{ticker}_MOM_ACCEL': momentum_acceleration
        })
    
    features_df = pd.concat(features_dict.values(), axis=1)
    features_df = features_df.dropna()
    print(f"技术特征计算完成: {features_df.shape[1]}个特征，{features_df.shape[0]}个样本")
    return features_df

# 在 local_preparation.py 中，修改 process_analyst_features 函数
def process_analyst_features(analyst_data, date_index, price_pivot):
    """最终版分析师特征处理 - 处理三个可用指标"""
    if analyst_data is None or analyst_data.empty:
        print("警告: analyst_data 为空")
        return pd.DataFrame()
    
    analyst_features = {}
    
    # 按股票代码处理
    for ticker in analyst_data['symbol'].unique():
        ticker_data = analyst_data[analyst_data['symbol'] == ticker].copy()
        
        # 设置日期索引并按日期排序
        ticker_data = ticker_data.set_index('date').sort_index()
        
        # 重采样到交易日（使用前向填充）
        eps_fy1_series = ticker_data['eps_fy1'].reindex(date_index, method='ffill') if 'eps_fy1' in ticker_data.columns else None
        eps_fy2_series = ticker_data['eps_fy2'].reindex(date_index, method='ffill') if 'eps_fy2' in ticker_data.columns else None
        rec_series = ticker_data['rec_mean'].reindex(date_index, method='ffill') if 'rec_mean' in ticker_data.columns else None
        
        feature_count = 0
        
        # EPS FY1 特征
        if eps_fy1_series is not None and not eps_fy1_series.isnull().all():
            # 1. EPS FY1 绝对值
            analyst_features[f'{ticker}_EPS_FY1'] = eps_fy1_series
            
            # 2. EPS FY1 30天变化率
            eps_fy1_change_30d = eps_fy1_series.pct_change(30)
            analyst_features[f'{ticker}_EPS_FY1_CHG_30D'] = eps_fy1_change_30d
            
            # 3. EPS FY1 90天变化率
            eps_fy1_change_90d = eps_fy1_series.pct_change(90)
            analyst_features[f'{ticker}_EPS_FY1_CHG_90D'] = eps_fy1_change_90d
            
            feature_count += 3
        
        # EPS FY2 特征
        if eps_fy2_series is not None and not eps_fy2_series.isnull().all():
            # 4. EPS FY2 绝对值
            analyst_features[f'{ticker}_EPS_FY2'] = eps_fy2_series
            
            # 5. EPS FY2 30天变化率
            eps_fy2_change_30d = eps_fy2_series.pct_change(30)
            analyst_features[f'{ticker}_EPS_FY2_CHG_30D'] = eps_fy2_change_30d
            
            feature_count += 2
        
        # 分析师推荐特征
        if rec_series is not None and not rec_series.isnull().all():
            # 6. 分析师推荐（1=强烈买入，5=强烈卖出）
            # 转换为买入信号：值越低越好，所以我们反转
            buy_signal = 5 - rec_series  # 反转，使值越高表示越看好
            analyst_features[f'{ticker}_REC_BUY_SIGNAL'] = buy_signal
            
            # 7. 推荐变化（负变化表示推荐变好）
            rec_change = -rec_series.diff(30)
            analyst_features[f'{ticker}_REC_CHG_30D'] = rec_change
            
            feature_count += 2
        
        # 跨期EPS特征（如果两个EPS都有）
        if (eps_fy1_series is not None and eps_fy2_series is not None and 
            not eps_fy1_series.isnull().all() and not eps_fy2_series.isnull().all()):
            # 8. FY2/FY1 EPS比率（增长预期）
            eps_growth_expectation = eps_fy2_series / eps_fy1_series - 1
            analyst_features[f'{ticker}_EPS_GROWTH_EXP'] = eps_growth_expectation
            
            # 9. FY2-FY1 EPS差异
            eps_diff = eps_fy2_series - eps_fy1_series
            analyst_features[f'{ticker}_EPS_DIFF'] = eps_diff
            
            feature_count += 2
        
        print(f"    {ticker}: 创建了 {feature_count} 个分析师特征")
    
    if not analyst_features:
        print("无法从分析师数据创建任何特征")
        return pd.DataFrame()
    
    # 创建DataFrame
    result_df = pd.DataFrame(analyst_features).reindex(date_index)
    
    print(f"✅ 分析师特征处理完成: 添加了 {len(analyst_features)} 个特征")
    
    return result_df
    

def create_labels_from_wrds(price_pivot, sp500_data, prediction_horizon):
    """创建标签：未来N日超额收益"""
    # 对齐标普500数据
    if 'date' in sp500_data.columns:
        sp500_data.set_index('date', inplace=True)
    
    # 确保索引是datetime
    sp500_data.index = pd.to_datetime(sp500_data.index)
    price_pivot.index = pd.to_datetime(price_pivot.index)
    
    if 'sp500_total_return' in sp500_data.columns:
        benchmark_prices = sp500_data['sp500_total_return']
    elif 'sp500_return' in sp500_data.columns:
        benchmark_prices = sp500_data['sp500_return']
    else:
        # 如果没有标普500数据，使用等权组合作为基准
        print("警告: 使用等权组合作为基准")
        benchmark_prices = price_pivot.mean(axis=1)
    
    # 计算未来超额收益
    labels_dict = {}
    
    for ticker in price_pivot.columns:
        stock_prices = price_pivot[ticker]
        
        # 对齐日期
        aligned_idx = stock_prices.index.intersection(benchmark_prices.index)
        stock_aligned = stock_prices[aligned_idx]
        bench_aligned = benchmark_prices[aligned_idx]
        
        # 计算未来累计超额收益
        # 股票未来收益
        stock_future = stock_aligned.shift(-prediction_horizon) / stock_aligned - 1
        
        # 基准未来收益
        bench_future = bench_aligned.shift(-prediction_horizon) / bench_aligned - 1
        
        # 超额收益
        excess_return = stock_future - bench_future
        labels_dict[f'{ticker}_EXCESS'] = excess_return
    
    labels_df = pd.DataFrame(labels_dict).dropna()
    print(f"标签创建完成: {labels_df.shape[1]}个标签")
    return labels_df

def align_features_labels(features, labels):
    """对齐特征和标签的日期索引"""
    common_idx = features.index.intersection(labels.index)
    aligned_features = features.loc[common_idx]
    aligned_labels = labels.loc[common_idx]
    
    print(f"数据对齐: {len(common_idx)}个共同日期")
    return aligned_features, aligned_labels

def prepare_datasets_with_proper_splits(features, labels):
    """正确处理数据划分"""
    print("\n处理数据划分...")
    
    # 确保有足够的样本
    if len(features) < 100:
        print(f"警告: 只有 {len(features)} 个样本，数据可能不足")
    
    # 按时间顺序划分（不要随机打乱！）
    # 60%训练，20%验证，20%测试
    split_idx1 = int(len(features) * 0.6)
    split_idx2 = int(len(features) * 0.8)
    
    # 使用索引而不是位置，确保日期连续性
    dates = features.index
    train_dates = dates[:split_idx1]
    val_dates = dates[split_idx1:split_idx2]
    test_dates = dates[split_idx2:]
    
    X_train = features.loc[train_dates]
    y_train = labels.loc[train_dates]
    X_val = features.loc[val_dates]
    y_val = labels.loc[val_dates]
    X_test = features.loc[test_dates]
    y_test = labels.loc[test_dates]
    
    print(f"训练集: {len(X_train)} 个样本 ({X_train.index.min().date()} 至 {X_train.index.max().date()})")
    print(f"验证集: {len(X_val)} 个样本 ({X_val.index.min().date()} 至 {X_val.index.max().date()})")
    print(f"测试集: {len(X_test)} 个样本 ({X_test.index.min().date()} 至 {X_test.index.max().date()})")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# 在 local_preparation.py 的 main 部分
if __name__ == '__main__':
    try:
        # 1. 获取数据
        features, labels, prices = prepare_and_save_datasets()
        
        # 2. 正确处理数据划分
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_datasets_with_proper_splits(features, labels)
        
        # 3. 保存划分信息
        output_dir = './data_for_colab'
        
        split_info = {
            'train_dates': X_train.index.strftime('%Y-%m-%d').tolist(),
            'val_dates': X_val.index.strftime('%Y-%m-%d').tolist(),
            'test_dates': X_test.index.strftime('%Y-%m-%d').tolist(),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_count': features.shape[1],
            'label_count': labels.shape[1]
        }
        
        with open(f'{output_dir}/split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"\n✅ 数据准备完成！")
        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        print(f"总特征数: {features.shape[1]}")
        
    except Exception as e:
        print(f"数据准备过程中出现错误: {e}")
        import traceback
        traceback.print_exc()