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
# from src.wrds_fetcher_v2 import WRDSFetcher
# from src.eikon_fetcher_v2 import EikonFetcher

# ==========================================
# 1. 长期趋势特征 (基于 OHLC 结构)
# ==========================================
def calculate_ohlc_ratios(df):
    """
    计算基于 OHLC 结构的长期预测特征
    """
    df = df.sort_values(['ticker', 'date'])
    
    # 1. 归一化波动率 (NATR)
    df['volatility_natr'] = (df['adj_high'] - df['adj_low']) / df['adj_prc']
    
    # 2. 收盘位置 (Close Location Value)
    denominator = df['adj_high'] - df['adj_low']
    df['close_loc'] = (df['adj_prc'] - df['adj_low']) / (denominator.replace(0, np.nan))
    
    # 3. 抛压 (上影线)
    real_body_top = df[['adj_open', 'adj_prc']].max(axis=1)
    df['shadow_upper'] = (df['adj_high'] - real_body_top) / df['adj_prc']
    
    # 4. 支撑 (下影线)
    real_body_bottom = df[['adj_open', 'adj_prc']].min(axis=1)
    df['shadow_lower'] = (real_body_bottom - df['adj_low']) / df['adj_prc']
    
    # 5. 实体大小
    df['body_size'] = (df['adj_prc'] - df['adj_open']) / df['adj_open']
    
    return df

# ==========================================
# 2. 滚动归一化 (Rolling Z-Score) - 时序特征
# ==========================================
def apply_rolling_normalization(df, window=60):
    """
    对宽表进行时序上的滚动归一化
    """
    print(f"正在进行时序滚动归一化 (Window={window})...")
    tickers = set([c.split('_')[0] for c in df.columns if '_' in c])
    normalized_dfs = []
    
    for ticker in tickers:
        cols = [c for c in df.columns if c.startswith(f"{ticker}_")]
        if not cols: continue
            
        sub_df = df[cols]
        roll_mean = sub_df.rolling(window=window).mean()
        roll_std = sub_df.rolling(window=window).std()
        
        z_score_df = (sub_df - roll_mean) / (roll_std + 1e-8)
        normalized_dfs.append(z_score_df)
    
    if not normalized_dfs: return df
    return pd.concat(normalized_dfs, axis=1)

# ==========================================
# [极致加速版] 3. 宽表转长表及截面标准化 
# ==========================================
# ==========================================
# [无前瞻偏差 + 极速版] 3. 宽表转长表及截面标准化 
# ==========================================
def convert_to_long_and_standardize(features_wide, labels_wide):
    """
    将宽表转换为 MultiIndex (date, ticker) 的长表，并进行截面标准化
    """
    print("\n[数据重塑] 开始将宽表转换为长表并进行截面标准化 (无前瞻偏差极速版)...")
    
    # [优化 1]：在长表展开前，提前将内存砍半，防止 stack 撑爆内存变卡
    float_cols = features_wide.select_dtypes(include=['float64']).columns
    features_wide[float_cols] = features_wide[float_cols].astype(np.float32)
    
    # --- A. 特征表 MultiIndex 转换 ---
    feat_cols = []
    for c in features_wide.columns:
        parts = c.split('_', 1)
        if len(parts) == 2:
            feat_cols.append((parts[0], parts[1]))
        else:
            feat_cols.append((c, 'UNKNOWN'))
            
    # 设置 MultiIndex 列 (Ticker, Feature_Name)
    features_wide.columns = pd.MultiIndex.from_tuples(feat_cols, names=['ticker', 'feature'])
    
    print("    -> 正在执行特征表 Stack 操作 (可能会吃点内存，请稍候)...")
    X_long = features_wide.stack(level='ticker').sort_index()
    
    # --- B. 标签表 MultiIndex 转换 (修复报错的关键点！) ---
    lab_cols = []
    for c in labels_wide.columns:
        ticker = c.split('_')[0]
        lab_cols.append((ticker, 'target'))
        
    # 必须给标签表也赋予 'ticker' 层级名称
    labels_wide.columns = pd.MultiIndex.from_tuples(lab_cols, names=['ticker', 'label_type'])
    
    print("    -> 正在执行标签表 Stack 操作...")
    y_long = labels_wide.stack(level='ticker').sort_index()
    
    # 提取单列 Series
    if isinstance(y_long, pd.DataFrame):
        y_long = y_long.iloc[:, 0]
        
    # ==========================================
    # 🌟 无前瞻偏差：先算特征截面，再做标签对齐！
    # ==========================================
    print("    -> [无偏差版] 正在执行特征截面 Z-Score 标准化 (包含当天所有存活股票)...")
    daily_mean = X_long.groupby(level=0).mean()
    daily_std = X_long.groupby(level=0).std()
    
    # 矩阵级别运算：减去均值，除以标准差
    X_final = X_long.sub(daily_mean, level=0).div(daily_std.replace(0, 1e-8), level=0)
    
    # 极值处理
    X_final = X_final.clip(lower=-3.0, upper=3.0).fillna(0) 

    # ==========================================
    # 🌟 特征算完后，再把没有未来收益的无效样本踢掉
    # ==========================================
    print("    -> 正在执行特征与标签的截面对齐...")
    common_idx = X_final.index.intersection(y_long.index)
    
    # 过滤特征和标签
    X_final = X_final.loc[common_idx]
    y_long_aligned = y_long.loc[common_idx]
    
    # --- E. 标签百分位 Rank (在对齐后的样本里进行排名) ---
    print("    -> 正在执行标签截面 Percentile Rank...")
    y_final = y_long_aligned.groupby(level=0).rank(pct=True)
    y_final = (y_final - 0.5) * 2.0
    y_final = y_final.fillna(0)
    
    # 稳妥转回 DataFrame
    y_final = y_final.to_frame(name='target_rank')
    
    print("    -> 数据重塑大功告成！")
    return X_final, y_final

def prepare_and_save_datasets():
    """本地数据准备主函数"""
    print("="*60)
    print("阶段一：本地数据准备 (整合WRDS + Eikon)")
    print("="*60)
    
    Config.setup_directories()
    
    print("\n[A] 从WRDS获取价格与基本面数据...")
    crsp_data=pd.read_parquet('./data/wrds_raw/fetch_crsp_data_dynamic.parquet')
    sp500_data=pd.read_parquet('./data/wrds_raw/sp500_index.parquet')
    funda_data=pd.read_parquet('./data/wrds_raw/fetch_compustat_fundamentals_dynamic.parquet')
    
    print("\n[B] 从本地Eikon获取IBES分析师预期数据...")
    analyst_data=pd.read_parquet('./data/eikon_raw/fetch_analyst_estimates_dynamic.parquet')
    
    print("\n[C] 特征工程 - 整合所有数据源...")

    crsp_data = crsp_data.replace([np.inf, -np.inf], np.nan).dropna()
    crsp_data = calculate_ohlc_ratios(crsp_data)
    
    price_pivot = crsp_data.pivot_table(index='date', columns='ticker', values='adj_prc')

    new_features_list = []
    ratio_cols = ['volatility_natr', 'close_loc', 'shadow_upper', 'shadow_lower', 'body_size', 'mkt_cap']

    for col in ratio_cols:
        if col in crsp_data.columns:
            pivoted = crsp_data.pivot_table(index='date', columns='ticker', values=col)
            pivoted.columns = [f"{ticker}_{col.upper()}" for ticker in pivoted.columns]
            new_features_list.append(pivoted)
            
    ohlc_features = pd.concat(new_features_list, axis=1) if new_features_list else pd.DataFrame()
    
    print("正在计算基础技术指标...")
    base_features = calculate_technical_features(price_pivot)

    feature_list = [base_features, ohlc_features]
    
    if analyst_data is not None and not analyst_data.empty:
        analyst_features = process_analyst_features(analyst_data, price_pivot.index, price_pivot)
        feature_list.append(analyst_features)
        print(f"已整合分析师特征: {len(analyst_features.columns)}个特征")
    else:
        print("警告: 未获取到分析师数据，仅使用技术特征")   
    
    if 'funda_data' in locals() and funda_data is not None and not funda_data.empty:
        print("正在处理 Compustat 基本面数据...")
        fundamental_features = process_fundamental_features(funda_data, price_pivot.index)
        feature_list.append(fundamental_features)
        print(f"已整合基本面特征: {fundamental_features.shape[1]} 个")
    
    all_features = pd.concat(feature_list, axis=1).sort_index()
    all_features = all_features.ffill()

    # Step 1: 时序滚动归一化
    all_features = apply_rolling_normalization(all_features, window=60)
    
    if len(all_features) > 60:
        all_features = all_features.iloc[60:]
    
    # all_features = all_features.fillna(0)
    
    print("\n[D] 创建预测标签...")
    labels = create_labels_from_wrds(price_pivot, sp500_data, Config.PREDICTION_HORIZON)
    
    print("\n[E] 对齐特征与标签...")
    aligned_features_wide, aligned_labels_wide = align_features_labels(all_features, labels)

    # ==============================================================
    # [关键修改] 调用新写的转换器，把宽表变为干净的长表面板数据
    # ==============================================================
    long_features, long_labels = convert_to_long_and_standardize(aligned_features_wide, aligned_labels_wide)
    
    print(f"\n数据统计:")
    print(f"- 面板特征维度: {long_features.shape[1]}个因子")
    print(f"- 截面样本总数: {long_features.shape[0]}行 (Date * Ticker)")
    dates_count = len(long_features.index.get_level_values(0).unique())
    print(f"- 交易日数量: {dates_count} 天")
    
    # F. 保存为标准化格式
    print("\n[F] 保存标准化数据文件...")
    output_dir = './data_for_colab'
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为 float32 节省内存
    float_cols = long_features.select_dtypes(include=['float64']).columns
    long_features[float_cols] = long_features[float_cols].astype(np.float32)

    # # Parquet 完美支持 MultiIndex 保存
    # long_features.to_parquet(f'{output_dir}/features_v2_60d_5d.parquet', compression='snappy')
    # long_labels.to_parquet(f'{output_dir}/labels_v2_60d_5d.parquet', compression='snappy')
    # price_pivot.to_parquet(f'{output_dir}/price_data_v2_60d_5d.parquet')

    # 使用变量动态生成文件名，避免 hard code
    horizon = Config.PREDICTION_HORIZON
    seq_len = Config.SEQUENCE_LENGTH
    
    print(f"\n[F] 正在保存 {horizon}d 预测周期的标准化数据文件...")
    
    # 动态构建文件名：例如 features_v2_60d_20d.parquet
    long_features.to_parquet(f'{output_dir}/features_v2_{seq_len}d_{horizon}d.parquet', compression='snappy')
    long_labels.to_parquet(f'{output_dir}/labels_v2_{seq_len}d_{horizon}d.parquet', compression='snappy')
    price_pivot.to_parquet(f'{output_dir}/price_data_v2_{seq_len}d_{horizon}d.parquet')
    
    print(f"✅ 文件已保存至: {output_dir}/features_v2_{seq_len}d_{horizon}d.parquet")
    
    meta_info = {
        'tickers': Config.TECH_TICKERS,
        'benchmark': Config.BENCHMARK_TICKER,
        'feature_columns': list(long_features.columns),
        'label_columns': list(long_labels.columns),
        'date_range': {
            'start': str(long_features.index.get_level_values(0).min().date()),
            'end': str(long_features.index.get_level_values(0).max().date())
        },
        'num_samples_total': len(long_features),
        'prediction_horizon': Config.PREDICTION_HORIZON,
        'sequence_length': Config.SEQUENCE_LENGTH,
        'preparation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f'{output_dir}/data_info.json', 'w') as f:
        json.dump(meta_info, f, indent=2)
    
    print("\n本地数据准备完成！已输出长表格式。")
    return long_features, long_labels, price_pivot

# 技术指标、基本面、分析师等计算函数保持你的原样逻辑
def calculate_technical_features(price_pivot):
    features_dict = {}
    if price_pivot.empty: return pd.DataFrame()

    for ticker in price_pivot.columns:
        price = price_pivot[ticker].dropna()
        if len(price) < 60: continue
        
        try:
            returns = price.pct_change()
            momentum = price.pct_change(20) 
            volatility = returns.rolling(20).std()
            
            delta = price.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            ma_60 = price.rolling(60).mean()
            price_ma_delta = (price - ma_60) / ma_60
            
            ma_short = price.rolling(10).mean()
            ma_long = price.rolling(50).mean()
            ma_ratio = ma_short / ma_long - 1
            
            bb_ma = price.rolling(20).mean()
            bb_std = price.rolling(20).std()
            bb_upper = bb_ma + 2 * bb_std
            bb_lower = bb_ma - 2 * bb_std
            bb_position = (price - bb_lower) / (bb_upper - bb_lower + 1e-9)
            
            high_20 = price.rolling(20).max()
            low_20 = price.rolling(20).min()
            channel_breakout = (price - high_20.shift(1)) / price
            channel_breakdown = (price - low_20.shift(1)) / price
            momentum_acceleration = momentum.diff(5)
            
            features_dict[ticker] = pd.DataFrame({
                f'{ticker}_mom': momentum, f'{ticker}_vol': volatility,
                f'{ticker}_rsi': rsi, f'{ticker}_ma_delta': price_ma_delta,
                f'{ticker}_ma_ratio': ma_ratio, f'{ticker}_bb_pos': bb_position,
                f'{ticker}_ch_breakout': channel_breakout, f'{ticker}_ch_breakdown': channel_breakdown,
                f'{ticker}_accel': momentum_acceleration
            }, index=price.index)
            
        except Exception as e: continue

    if not features_dict: return pd.DataFrame(index=price_pivot.index)
    features_df = pd.concat(features_dict.values(), axis=1)
    return features_df.reindex(price_pivot.index)

def process_fundamental_features(funda_df, daily_index):
    df = funda_df.copy()
    df['rdq'] = pd.to_datetime(df['rdq'])
    df['datadate'] = pd.to_datetime(df['datadate'])
    df['effective_date'] = df['rdq'].fillna(df['datadate'] + pd.Timedelta(days=90))
    
    if 'ltq' in df.columns and 'atq' in df.columns:
        df['debt_to_asset'] = df['ltq'] / df['atq']
    if 'niq' in df.columns and 'saleq' in df.columns:
        df['net_margin'] = df['niq'] / df['saleq']
    if 'niq' in df.columns and 'seqq' in df.columns:
        df['roe'] = df['niq'] / df['seqq']

    feature_cols = ['debt_to_asset', 'net_margin', 'roe', 'epsfxq']
    valid_cols = [c for c in feature_cols if c in df.columns]
    
    pivot_list = []
    for col in valid_cols:
        pivoted = df.pivot_table(index='effective_date', columns='ticker', values=col, aggfunc='last')
        pivoted.columns = [f"{ticker}_{col.upper()}" for ticker in pivoted.columns]
        pivot_list.append(pivoted)
        
    if not pivot_list: return pd.DataFrame()
    funda_wide = pd.concat(pivot_list, axis=1)
    return funda_wide.reindex(daily_index).ffill()

def process_analyst_features(analyst_data, date_index, price_pivot):
    if analyst_data is None or analyst_data.empty: return pd.DataFrame()
    
    analyst_features = {}
    for ticker in analyst_data['symbol'].unique():
        ticker_data = analyst_data[analyst_data['symbol'] == ticker].copy()
        ticker_data = ticker_data.set_index('date').sort_index()
        
        eps_fy1_series = ticker_data['eps_fy1'].reindex(date_index, method='ffill') if 'eps_fy1' in ticker_data.columns else None
        eps_fy2_series = ticker_data['eps_fy2'].reindex(date_index, method='ffill') if 'eps_fy2' in ticker_data.columns else None
        rec_series = ticker_data['rec_mean'].reindex(date_index, method='ffill') if 'rec_mean' in ticker_data.columns else None
        
        if eps_fy1_series is not None and not eps_fy1_series.isnull().all():
            analyst_features[f'{ticker}_EPS_FY1'] = eps_fy1_series
            analyst_features[f'{ticker}_EPS_FY1_CHG_30D'] = eps_fy1_series.pct_change(30)
            analyst_features[f'{ticker}_EPS_FY1_CHG_90D'] = eps_fy1_series.pct_change(90)
        
        if eps_fy2_series is not None and not eps_fy2_series.isnull().all():
            analyst_features[f'{ticker}_EPS_FY2'] = eps_fy2_series
            analyst_features[f'{ticker}_EPS_FY2_CHG_30D'] = eps_fy2_series.pct_change(30)
        
        if rec_series is not None and not rec_series.isnull().all():
            analyst_features[f'{ticker}_REC_BUY_SIGNAL'] = 5 - rec_series
            analyst_features[f'{ticker}_REC_CHG_30D'] = -rec_series.diff(30)
        
        if (eps_fy1_series is not None and eps_fy2_series is not None and 
            not eps_fy1_series.isnull().all() and not eps_fy2_series.isnull().all()):
            analyst_features[f'{ticker}_EPS_GROWTH_EXP'] = eps_fy2_series / eps_fy1_series - 1
            analyst_features[f'{ticker}_EPS_DIFF'] = eps_fy2_series - eps_fy1_series
    
    if not analyst_features: return pd.DataFrame()
    return pd.DataFrame(analyst_features).reindex(date_index)
    
def create_labels_from_wrds(price_pivot, sp500_data, prediction_horizon):
    if 'date' in sp500_data.columns:
        sp500_data = sp500_data.set_index('date')
    
    sp500_data.index = pd.to_datetime(sp500_data.index)
    price_pivot.index = pd.to_datetime(price_pivot.index)
    
    if 'sp500_total_return' in sp500_data.columns:
        benchmark_prices = (1 + sp500_data['sp500_total_return']).cumprod()
    else:
        benchmark_prices = price_pivot.mean(axis=1).pct_change().fillna(0).add(1).cumprod()

    stock_future_ret = price_pivot.shift(-prediction_horizon) / price_pivot - 1
    bench_future_ret = benchmark_prices.shift(-prediction_horizon) / benchmark_prices - 1
    
    labels_df = stock_future_ret.sub(bench_future_ret, axis=0)
    labels_df.columns = [f'{col}_EXCESS' for col in labels_df.columns]
    
    os.makedirs('./data', exist_ok=True)
    labels_df.to_csv(f'./data/labels_excess_return.csv')
    return labels_df

def align_features_labels(features, labels):
    common_dates = features.index.intersection(labels.index)
    aligned_X = features.loc[common_dates]
    aligned_y = labels.loc[common_dates]
    
    valid_mask = aligned_y.notna().any(axis=1)
    aligned_X = aligned_X[valid_mask]
    aligned_y = aligned_y[valid_mask]
    
    return aligned_X, aligned_y

def prepare_datasets_with_proper_splits(long_features, long_labels):
    """
    修改为基于长表面板数据的划分逻辑 (按时间日期切分)
    使用 .isin() 完美解决 MultiIndex 的 KeyError 问题
    """
    print("\n处理数据时间切分...")
    
    # 提取唯一的日期并排序
    dates = long_features.index.get_level_values(0).unique().sort_values()
    
    split_idx1 = int(len(dates) * 0.6)
    split_idx2 = int(len(dates) * 0.8)
    
    train_dates = dates[:split_idx1]
    val_dates = dates[split_idx1:split_idx2]
    test_dates = dates[split_idx2:]
    
    # 【核心修复】：放弃脆弱的 .loc[] 列表切片，改用稳如磐石的 .isin() 布尔掩码
    train_mask = long_features.index.get_level_values(0).isin(train_dates)
    val_mask = long_features.index.get_level_values(0).isin(val_dates)
    test_mask = long_features.index.get_level_values(0).isin(test_dates)
    
    X_train = long_features[train_mask]
    y_train = long_labels[train_mask]
    
    X_val = long_features[val_mask]
    y_val = long_labels[val_mask]
    
    X_test = long_features[test_mask]
    y_test = long_labels[test_mask]
    
    print(f"训练集: {len(X_train)} 个样本 (日期范围: {train_dates.min().date()} 至 {train_dates.max().date()})")
    print(f"验证集: {len(X_val)} 个样本 (日期范围: {val_dates.min().date()} 至 {val_dates.max().date()})")
    print(f"测试集: {len(X_test)} 个样本 (日期范围: {test_dates.min().date()} 至 {test_dates.max().date()})")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == '__main__':
    try:
        # 1. 获取并处理数据
        features, labels, prices = prepare_and_save_datasets()
        
        # 2. 正确处理数据划分
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_datasets_with_proper_splits(features, labels)
        
        # 3. 保存划分信息
        output_dir = './data_for_colab'
        
        split_info = {
            'train_dates': [str(d.date()) for d in X_train.index.get_level_values(0).unique()],
            'val_dates': [str(d.date()) for d in X_val.index.get_level_values(0).unique()],
            'test_dates': [str(d.date()) for d in X_test.index.get_level_values(0).unique()],
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_count': features.shape[1],
            'label_count': labels.shape[1]
        }
        
        with open(f'{output_dir}/split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"\n✅ 数据准备全部完成！现在可以直接去构建 Dataset 了。")
        
    except Exception as e:
        print(f"数据准备过程中出现错误: {e}")
        import traceback
        traceback.print_exc()