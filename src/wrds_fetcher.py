import wrds
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
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
    
    def fetch_crsp_data(self, symbols, start_date='2018-01-01', end_date='2023-12-31'):
        """
        从CRSP获取股票价格和交易数据
        参数:
            symbols: 股票代码列表（需要是CRSP格式的PERMNO或TICKER）
            建议使用TICKER，如 ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
        返回:
            包含调整后价格、成交量等数据的DataFrame
        """
        # 将股票代码转换为字符串格式用于SQL查询
        ticker_list = ", ".join([f"'{ticker}'" for ticker in symbols])
        
        print(f"从CRSP获取价格数据 ({start_date} 至 {end_date})...")
        
        # CRSP核心查询 - 获取调整后价格
        # 注意：这里使用dsf（日度股票文件）表，包含调整因子
        sql = f"""
        SELECT 
            a.date,
            a.permno,
            a.permco,
            a.cusip,
            b.ticker,
            a.ret,           -- 日收益率（已调整）
            a.retx,          -- 不考虑股息的收益率
            a.prc,           -- 收盘价
            a.askhi,         -- 当日最高价
            a.bidlo,         -- 当日最低价
            a.openprc,       -- 开盘价
            a.vol,           -- 交易量
            a.shrout,        -- 流通股数
            a.cfacpr,        -- 价格调整因子
            a.cfacshr        -- 流通股调整因子
        FROM 
            crsp.dsf AS a
        JOIN 
            crsp.dsenames AS b ON a.permno = b.permno
        WHERE 
            b.ticker IN ({ticker_list})
            AND b.namedt <= a.date
            AND (b.nameendt >= a.date OR b.nameendt IS NULL)
            AND a.date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY 
            b.ticker, a.date
        """
        
        df = self.db.raw_sql(sql, date_cols=['date'])
        
        # 计算调整后收盘价
        if 'prc' in df.columns and 'cfacpr' in df.columns:
            # prc字段在CRSP中有时为负值（表示交易特征），取绝对值
            df['adj_prc'] = np.abs(df['prc']) / df['cfacpr']
        
        print(f"获取到 {len(df)} 行CRSP数据，涵盖 {df['ticker'].nunique()} 只股票")
        return df
    
    def fetch_compustat_fundamentals(self, symbols, start_date='2018-01-01', end_date='2023-12-31'):
        """
        从Compustat获取季度基本面数据
        参数:
            symbols: 股票代码列表（Compustat GVKEY或TICKER）
        返回:
            季度基本面数据DataFrame
        """
        ticker_list = ", ".join([f"'{ticker}'" for ticker in symbols])
        
        print("从Compustat获取基本面数据...")
        
        # 获取季度基本面数据
        sql = f"""
        SELECT 
            gvkey,
            datadate,
            fyearq,
            fqtr,
            tic as ticker,
            saleq,          -- 季度销售额
            revtq,          -- 季度总收入
            niq,            -- 季度净利润
            atq,            -- 季度总资产
            seqq,           -- 季度股东权益
            ceqq,           -- 普通股权益
            dlttq,          -- 长期债务
            actq,           -- 流动资产
            lctq,           -- 流动负债
            epsfxq,         -- 稀释后EPS
            prccq,          -- 季度收盘价
            cshoq           -- 季度流通股数
        FROM 
            comp.fundq
        WHERE 
            tic IN ({ticker_list})
            AND datadate BETWEEN '{start_date}' AND '{end_date}'
            AND indfmt = 'INDL'  -- 工业格式
            AND datafmt = 'STD'  -- 标准格式
            AND consol = 'C'     -- 合并报表
            AND popsrc = 'D'     -- 主要来源
        ORDER BY 
            tic, datadate
        """
        
        funda_df = self.db.raw_sql(sql, date_cols=['datadate'])
        
        # 计算常用财务比率
        if not funda_df.empty:
            # 市净率 (P/B) - 需要市值数据，这里简化处理
            if 'seqq' in funda_df.columns and 'cshoq' in funda_df.columns and 'prccq' in funda_df.columns:
                funda_df['pb_ratio'] = (funda_df['prccq'] * funda_df['cshoq']) / funda_df['seqq']
            
            # 资产负债率
            if 'dlttq' in funda_df.columns and 'atq' in funda_df.columns:
                funda_df['debt_ratio'] = funda_df['dlttq'] / funda_df['atq']
        
        print(f"获取到 {len(funda_df)} 行Compustat基本面数据")
        return funda_df
    
    def fetch_ibes_estimates(self, symbols, start_date='2018-01-01', end_date='2023-12-31'):
        """
        从IBES获取分析师预期数据
        """
        ticker_list = ", ".join([f"'{ticker}'" for ticker in symbols])
        
        print("从IBES获取分析师预期数据...")
        
        # 获取EPS预测摘要
        sql = f"""
        SELECT 
            ticker,
            statpers as estimate_date,
            fpedats as period_end_date,
            medest as median_eps,      -- 中位数EPS预测
            meanest as mean_eps,       -- 平均EPS预测
            numest as num_analysts,    -- 分析师数量
            stdev                     -- 预测标准差
        FROM 
            ibes.statsum_epsus
        WHERE 
            ticker IN ({ticker_list})
            AND statpers BETWEEN '{start_date}' AND '{end_date}'
            AND fiscalp = 'ANN'        -- 年度预测
        ORDER BY 
            ticker, statpers
        """
        
        ibes_df = self.db.raw_sql(sql, date_cols=['estimate_date', 'period_end_date'])
        
        print(f"获取到 {len(ibes_df)} 行IBES预期数据")
        return ibes_df
    
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
    # 初始化连接（首次会提示输入密码）
    fetcher = WRDSFetcher(wrds_username='your_username')
    
    # 定义股票列表（使用CRSP中的TICKER）
    tech_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
    
    # 获取数据
    price_data = fetcher.fetch_crsp_data(tech_tickers)
    funda_data = fetcher.fetch_compustat_fundamentals(tech_tickers)
    ibes_data = fetcher.fetch_ibes_estimates(tech_tickers)
    sp500_data = fetcher.fetch_sp500_index()
    
    # 保存数据
    price_data.to_parquet('./data/wrds_raw/crsp_prices.parquet')
    funda_data.to_parquet('./data/wrds_raw/compustat_fundamentals.parquet')
    ibes_data.to_parquet('./data/wrds_raw/ibes_estimates.parquet')
    sp500_data.to_parquet('./data/wrds_raw/sp500_index.parquet')
    
    fetcher.close()