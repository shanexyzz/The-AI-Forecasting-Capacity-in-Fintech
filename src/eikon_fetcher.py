# src/eikon_fetcher.py
import eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EikonFetcher:
    """
    使用Eikon作为数据补充（主要用于获取IBES分析师数据）
    """
    def __init__(self, app_key=None):
        """
        初始化Eikon连接
        参数:
            app_key: Eikon App Key，如果为None会尝试从环境变量读取
        """
        if app_key:
            ek.set_app_key(app_key)
        else:
            # 尝试从环境变量读取
            import os
            app_key = os.getenv('EIKON_APP_KEY')
            if app_key:
                ek.set_app_key(app_key)
            else:
                print("警告: 未设置Eikon App Key")
    
    # 在 src/eikon_fetcher.py 中，完全重写 fetch_analyst_estimates 方法
    def fetch_analyst_estimates(self, symbols, start_date='2018-01-01', end_date='2023-12-31'):
        """
        最终版：获取三个可用的分析师指标
        """
        print(f"从Eikon获取IBES分析师数据 ({start_date} 至 {end_date})...")
        
        all_stock_data = []
        
        for symbol in symbols:
            eikon_symbol = f"{symbol}.O"
            
            try:
                print(f"\n  处理 {symbol}...")
                
                # 1. 获取FY1 EPS数据
                print(f"    获取FY1 EPS数据...")
                df_fy1, err1 = ek.get_data(
                    eikon_symbol, 
                    ['TR.EPSMean(Period=FY1).calcdate', 'TR.EPSMean(Period=FY1)'],
                    {'SDate': start_date, 'EDate': end_date, 'Frq': 'D'}
                )
                
                # 2. 获取FY2 EPS数据
                print(f"    获取FY2 EPS数据...")
                df_fy2, err2 = ek.get_data(
                    eikon_symbol, 
                    ['TR.EPSMean(Period=FY2).calcdate', 'TR.EPSMean(Period=FY2)'],
                    {'SDate': start_date, 'EDate': end_date, 'Frq': 'D'}
                )
                
                # 3. 获取推荐数据（使用可用的TR.RecMean）
                print(f"    获取推荐数据...")
                df_rec, err3 = ek.get_data(
                    eikon_symbol, 
                    ['TR.RecMean.calcdate', 'TR.RecMean'],
                    {'SDate': start_date, 'EDate': end_date, 'Frq': 'D'}
                )
                
                # 检查错误
                if err1:
                    print(f"    FY1 EPS错误: {err1}")
                if err2:
                    print(f"    FY2 EPS错误: {err2}")
                if err3:
                    print(f"    推荐数据错误: {err3}")
                
                # 初始化数据字典
                data_dict = {}
                
                # 处理FY1数据
                if df_fy1 is not None and not df_fy1.empty:
                    data_dict['eps_fy1'] = df_fy1.rename(columns={
                        'Calc Date': 'date',
                        'Earnings Per Share - Mean': 'eps_fy1'
                    })[['date', 'eps_fy1']]
                    print(f"    FY1 EPS: {len(data_dict['eps_fy1'])} 行")
                
                # 处理FY2数据
                if df_fy2 is not None and not df_fy2.empty:
                    data_dict['eps_fy2'] = df_fy2.rename(columns={
                        'Calc Date': 'date',
                        'Earnings Per Share - Mean': 'eps_fy2'
                    })[['date', 'eps_fy2']]
                    print(f"    FY2 EPS: {len(data_dict['eps_fy2'])} 行")
                
                # 处理推荐数据
                if df_rec is not None and not df_rec.empty:
                    # 查找正确的列名
                    rec_value_col = None
                    date_col = None
                    
                    for col in df_rec.columns:
                        if 'Rec' in str(col) and 'Date' not in str(col):
                            rec_value_col = col
                        elif 'Date' in str(col):
                            date_col = col
                    
                    if rec_value_col and date_col:
                        data_dict['rec_mean'] = df_rec.rename(columns={
                            date_col: 'date',
                            rec_value_col: 'rec_mean'
                        })[['date', 'rec_mean']]
                        print(f"    推荐数据: {len(data_dict['rec_mean'])} 行")
                    else:
                        print(f"    无法识别推荐数据列: {df_rec.columns.tolist()}")
                
                if not data_dict:
                    print(f"    {symbol}: 无有效数据")
                    continue
                
                # 合并所有指标
                merged_df = None
                for key, df in data_dict.items():
                    df['date'] = pd.to_datetime(df['date'])
                    df['symbol'] = symbol
                    
                    if merged_df is None:
                        merged_df = df
                    else:
                        merged_df = pd.merge(merged_df, df, on=['date', 'symbol'], how='outer')
                
                # 排序并前向填充
                merged_df = merged_df.sort_values('date')
                merged_df = merged_df.fillna(method='ffill')
                
                # 移除完全重复的行
                merged_df = merged_df.drop_duplicates()
                
                all_stock_data.append(merged_df)
                print(f"    ✅ {symbol} 合并完成: {merged_df.shape}")
                print(f"      可用列: {[col for col in merged_df.columns if col not in ['date', 'symbol']]}")
                    
            except Exception as e:
                print(f"    处理 {symbol} 时异常: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_stock_data:
            print("警告: 未获取到任何分析师数据")
            return pd.DataFrame()
        
        # 合并所有股票的数据
        combined_df = pd.concat(all_stock_data, ignore_index=True)
        
        print(f"\n" + "="*60)
        print("✅ 成功获取分析师数据:")
        print("="*60)
        print(f"总数据量: {len(combined_df)} 行")
        print(f"股票数量: {combined_df['symbol'].nunique()}")
        print(f"日期范围: {combined_df['date'].min().date()} 到 {combined_df['date'].max().date()}")
        
        # 统计每个指标
        indicators = [col for col in combined_df.columns if col not in ['date', 'symbol']]
        print(f"\n可用指标 ({len(indicators)}):")
        for indicator in indicators:
            non_null = combined_df[indicator].notnull().sum()
            print(f"  {indicator}: {non_null} 个非空值 ({non_null/len(combined_df):.1%})")
        
        if not combined_df.empty:
            print(f"\n数据示例:")
            print(combined_df.head())
        
        return combined_df
            
    def fetch_supplemental_price_data(self, symbols, start_date='2018-01-01', end_date='2023-12-31'):
        """
        从Eikon获取补充价格数据（当WRDS数据有缺失时使用）
        """
        print(f"从Eikon获取补充价格数据...")
        
        # Eikon需要特定的RIC格式
        eikon_symbols = [f"{sym}.O" for sym in symbols]
        
        # 价格数据字段
        price_fields = ['TR.PriceClose', 'TR.PriceOpen', 'TR.PriceHigh', 'TR.PriceLow', 'TR.Volume']
        
        try:
            data, err = ek.get_data(
                eikon_symbols,
                price_fields,
                {'SDate': start_date, 'EDate': end_date}
            )
            
            if err:
                print(f"Eikon获取数据错误: {err}")
                return pd.DataFrame()
            
            if data is not None:
                # 处理返回的数据
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
                print(f"从Eikon获取到 {len(data)} 行价格数据")
                return data
                
        except Exception as e:
            print(f"Eikon API异常: {e}")
            return pd.DataFrame()
    
    def fetch_news_sentiment(self, symbols, start_date='2022-01-01', end_date='2022-12-31'):
        """
        获取新闻情绪数据（示例，可根据需要扩展）
        """
        # 这里只是示例，实际使用时需要根据Eikon的新闻API调整
        print("注意: 新闻情绪数据获取需要特定的Eikon新闻API权限")
        return pd.DataFrame()

# 使用示例
if __name__ == '__main__':
    # 设置你的Eikon App Key
    eikon_fetcher = EikonFetcher(app_key='your_eikon_app_key')
    
    tech_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
    
    # 测试获取分析师数据
    analyst_data = eikon_fetcher.fetch_analyst_estimates(
        tech_tickers,
        start_date='2022-01-01',
        end_date='2022-12-31'
    )
    
    if not analyst_data.empty:
        print(f"分析师数据示例:")
        print(analyst_data.head())
        print(f"\n数据类型:")
        print(analyst_data.dtypes)
        
        # 保存数据
        import os
        os.makedirs('./data/eikon_raw', exist_ok=True)
        analyst_data.to_parquet('./data/eikon_raw/analyst_data.parquet')
        print(f"\n数据已保存至 ./data/eikon_raw/analyst_data.parquet")