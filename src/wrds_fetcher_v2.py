import wrds
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os  # 修复：添加缺失的库
warnings.filterwarnings('ignore')

class WRDSFetcher:
    """
    从WRDS获取美股核心数据（CRSP, Compustat, IBES）
    """
    def __init__(self, wrds_username=None):
        """
        初始化WRDS连接
        参数:
            wrds_username: 你的WRDS用户名，如果为None，会尝试从环境变量读取
        """
        print("正在连接WRDS数据库...")
        self.db = wrds.Connection(wrds_username=wrds_username)
        print("WRDS连接成功!")
    
    def fetch_sp500_constituents(self, start_date='2018-01-01', end_date='2023-12-31'):
        """
        获取历史 S&P 500 成分股名单及其在榜时间段
        """
        print(f"正在获取 S&P 500 历史成分股变动记录...")
        
        # 从 crsp.msp500list 获取 permno 以及入选/剔除日期
        sql = f"""
        SELECT permno, start, resend as end
        FROM crsp.msp500list
        WHERE (resend >= '{start_date}' OR resend IS NULL)
          AND start <= '{end_date}'
        """
        sp500_list = self.db.raw_sql(sql, date_cols=['start', 'end'])
        
        # 将无结束日期的记录填充为当前日期（表示目前仍在榜）
        sp500_list['end'] = sp500_list['end'].fillna(pd.to_datetime('today'))
        return sp500_list

    def fetch_crsp_data_dynamic(self, start_date='2018-01-01', end_date='2023-12-31'):
        """
        自动根据 S&P 500 历史成分股动态获取价格数据
        """
        # 1. 先拿到成分股的 PERMNO 列表
        constituents = self.fetch_sp500_constituents(start_date, end_date)
        permno_list = tuple(constituents['permno'].unique().tolist())
        
        print(f"在 {start_date} 到 {end_date} 期间，共有 {len(permno_list)} 只股票曾进入 S&P 500")

        # 2. 从 CRSP 获取这些股票的所有历史数据
        sql = f"""
        SELECT 
            a.date, b.ticker, a.permno,
            a.ret, a.prc, a.cfacpr
        FROM 
            crsp.dsf AS a
        JOIN 
            crsp.dsenames AS b ON a.permno = b.permno
        WHERE 
            a.permno IN {permno_list}
            AND b.namedt <= a.date
            AND (b.nameendt >= a.date OR b.nameendt IS NULL)
            AND a.date BETWEEN '{start_date}' AND '{end_date}'
        """
        raw_prices = self.db.raw_sql(sql, date_cols=['date'])
        
        # 3. 关键步骤：过滤掉不在当期成分股名单内的记录
        # 我们将价格表与成分股变动表按 permno 进行关联
        merged = pd.merge(raw_prices, constituents, on='permno')
        
        # 只保留 [start, end] 时间段内的记录
        dynamic_df = merged[(merged['date'] >= merged['start']) & (merged['date'] <= merged['end'])]
        
        # 清洗掉不需要的列
        dynamic_df = dynamic_df.drop(columns=['start', 'end']).sort_values(['ticker', 'date'])
        
        if not dynamic_df.empty:
            dynamic_df['adj_prc'] = np.abs(dynamic_df['prc']) / dynamic_df['cfacpr']
            
        print(f"过滤完成：获取到 {len(dynamic_df)} 行动态成分股数据")
        return dynamic_df
    
    def fetch_compustat_fundamentals_dynamic(self, start_date='2018-01-01', end_date='2023-12-31'):
        """
        根据 S&P 500 动态成分股获取 Compustat 季度财务数据
        利用 CCM 链接表确保 permno 与 gvkey 严格匹配
        """
        # 1. 首先获取 S&P 500 历史成分股的 permno 列表
        constituents = self.fetch_sp500_constituents(start_date, end_date)
        permno_list = tuple(constituents['permno'].unique().tolist())
        
        print(f"正在通过 CCM 链接表匹配并获取 Compustat 基本面数据...")

        # 2. 核心 SQL：通过 ccmxpf_linktable 关联 permno 和 gvkey
        # linktype 在 ('LU', 'LC') 确保了链接的高质量
        sql = f"""
        SELECT 
            l.permno, f.gvkey, f.datadate, f.rdq,
            f.tic as ticker, f.fyearq, f.fqtr,
            f.saleq, f.niq, f.atq, f.seqq, f.ltq, f.epsfxq
        FROM 
            comp.fundq AS f
        JOIN 
            crsp.ccmxpf_linktable AS l ON f.gvkey = l.gvkey
        WHERE 
            l.permno IN {permno_list}
            AND l.linktype IN ('LU', 'LC')
            AND l.linkprim IN ('P', 'C')
            AND f.datadate BETWEEN l.linkdt AND COALESCE(l.linkenddt, CURRENT_DATE)
            AND f.datadate BETWEEN '{start_date}' AND '{end_date}'
            AND f.indfmt = 'INDL' AND f.datafmt = 'STD' AND f.consol = 'C'
        ORDER BY 
            l.permno, f.datadate
        """
        
        funda_df = self.db.raw_sql(sql, date_cols=['datadate', 'rdq'])

        # 3. 再次进行动态成分股过滤
        # 确保财务报表日期（datadate）在公司属于 S&P 500 的时间段内
        merged = pd.merge(funda_df, constituents, on='permno')
        dynamic_funda = merged[
            (merged['datadate'] >= merged['start']) & 
            (merged['datadate'] <= merged['end'])
        ]
        
        # 移除辅助列并计算财务指标
        dynamic_funda = dynamic_funda.drop(columns=['start', 'end'])
        if not dynamic_funda.empty:
            # 例如计算 ROE (季度净利润/股东权益)
            dynamic_funda['roe'] = dynamic_funda['niq'] / dynamic_funda['seqq']
            
        print(f"获取到 {len(dynamic_funda)} 行动态 Compustat 数据")
        return dynamic_funda
    
    def fetch_ibes_estimates_dynamic(self, start_date='2018-01-01', end_date='2023-12-31'):
        """
        根据 S&P 500 动态成分股获取 IBES 分析师一致预期数据
        利用 wrds.ibclink 链接表确保 IBES Ticker 与 CRSP permno 严格对齐
        """
        # 1. 获取 S&P 500 历史成分股的 permno 列表
        constituents = self.fetch_sp500_constituents(start_date, end_date)
        permno_list = tuple(constituents['permno'].unique().tolist())
        
        print(f"正在通过 IBES/CRSP 链接表匹配预测数据...")

        # 2. 核心 SQL：通过 wrds.ibclink 关联 permno 和 ibes ticker
        # ibes.statsum_epsus 包含分析师对 EPS 的中位数、平均值等预测汇总
        sql = f"""
        SELECT 
            ic.permno, 
            s.ticker as ibes_ticker, 
            s.statpers as estimate_date, 
            s.fpedats as period_end_date,
            s.medest as median_eps, 
            s.meanest as mean_eps, 
            s.numest as num_analysts,
            s.stdev as est_std
        FROM 
            ibes.statsum_epsus AS s
        JOIN 
            wrds.ibclink AS ic ON s.ticker = ic.ticker
        WHERE 
            ic.permno IN {permno_list}
            AND s.statpers BETWEEN ic.sdate AND COALESCE(ic.edate, CURRENT_DATE)
            AND s.statpers BETWEEN '{start_date}' AND '{end_date}'
            AND s.fiscalp = 'ANN'  -- 获取年度 EPS 预测
        ORDER BY 
            ic.permno, s.statpers
        """
        
        ibes_df = self.db.raw_sql(sql, date_cols=['estimate_date', 'period_end_date'])

        # 3. 动态成分股时间过滤
        # 确保预测日期 (estimate_date) 时，该股票确实在 S&P 500 指数内
        merged = pd.merge(ibes_df, constituents, on='permno')
        dynamic_ibes = merged[
            (merged['estimate_date'] >= merged['start']) & 
            (merged['estimate_date'] <= merged['end'])
        ]
        
        # 移除辅助列
        dynamic_ibes = dynamic_ibes.drop(columns=['start', 'end'])
        
        print(f"获取到 {len(dynamic_ibes)} 行动态 IBES 预测数据")
        return dynamic_ibes

    def close(self):
        self.db.close()
        print("WRDS连接已关闭")
    
    def fetch_sp500_index(self, start_date='2018-01-01', end_date='2023-12-31'):
        """
        获取标普500指数数据作为基准
        """
        print("获取标普500指数数据...")
        
        sql = f"""
        SELECT 
            date,
            vwretd as sp500_return,   -- 市值加权回报率
            sprtrn as sp500_total_return  -- 总回报率指数
        FROM 
            crsp.dsi
        WHERE 
            date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY 
            date
        """
        
        sp500_df = self.db.raw_sql(sql, date_cols=['date'])
        return sp500_df
    
    def close(self):
        """关闭数据库连接"""
        self.db.close()
        print("WRDS连接已关闭")

# 使用示例
if __name__ == '__main__':
    # Create the directory if it doesn't exist
    output_path = './data/wrds_raw'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")
    # 初始化连接（首次会提示输入密码）
    fetcher = WRDSFetcher(wrds_username='your_username')
    
    # 定义股票列表（使用CRSP中的TICKER）
    tech_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA','FB']
    
    # 获取数据
    price_data = fetcher.fetch_crsp_data(tech_tickers)
    funda_data = fetcher.fetch_compustat_fundamentals(tech_tickers)
    ibes_data = fetcher.fetch_ibes_estimates(tech_tickers)
    sp500_data = fetcher.fetch_sp500_index()
    
    '''
    # 保存数据
    price_data.to_parquet('./data/wrds_raw/crsp_prices.parquet')
    funda_data.to_parquet('./data/wrds_raw/compustat_fundamentals.parquet')
    ibes_data.to_parquet('./data/wrds_raw/ibes_estimates.parquet')
    sp500_data.to_parquet('./data/wrds_raw/sp500_index.parquet')
    '''
    files = {
        "crsp_prices.parquet": price_data,
        "compustat_fundamentals.parquet": funda_data,
        "ibes_estimates.parquet": ibes_data,
        "sp500_index.parquet": sp500_data
    }

    for filename, df in files.items():
        if df is not None and not df.empty:
            full_file_path = os.path.join(output_path, filename)
            df.to_parquet(full_file_path)
            print(f"Successfully saved: {full_file_path} ({len(df)} rows)")
        else:
            print(f"Warning: {filename} is empty and was not saved.")
    
    fetcher.close()