# src/config.py
# 项目配置参数

class Config:
    """配置类，存储所有项目参数"""
    
    # 1. 股票列表
    # 核心科技股FAAMG + NVDA + TSLA + 基准SPY
    TECH_TICKERS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
    
    # 基准股票代码 (用于计算超额收益)
    BENCHMARK_TICKER = 'SPY'
    
    # 2. 时间范围
    START_DATE = '2018-01-01'
    END_DATE = '2023-12-31'
    
    
    # 数据集划分日期
    TRAIN_END_DATE = '2022-06-30'    # 训练集结束：2022年6月30日
    VAL_END_DATE = '2022-12-31'      # 验证集结束：2022年12月31日
    # 测试集: 2022-01-01 至 END_DATE
    
    # 3. 特征参数
    SEQUENCE_LENGTH = 60  # 输入序列长度（过去60个交易日）
    PREDICTION_HORIZON = 5  # 预测未来5个交易日
    
    # 4. 模型参数
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    
    # Transformer架构参数
    D_MODEL = 64  # 模型维度
    NHEAD = 4     # 注意力头数
    NUM_LAYERS = 2  # Transformer层数
    DROPOUT = 0.1   # Dropout率
    
    # 5. 回测参数
    INITIAL_CAPITAL = 10000  # 初始资金
    TOP_N = 3  # 每周买入前N只股票
    REBALANCE_FREQ = 'W'  # 调仓频率：每周 ('W')
    
    # 6. 文件路径
    DATA_DIR = './data'
    RAW_DATA_DIR = './data/raw'
    PROCESSED_DATA_DIR = './data/processed'
    MODEL_DIR = './models'
    RESULTS_DIR = './results'
    
    # 7. API配置 (这些敏感信息建议从环境变量读取)
    # WRDS用户名 (通过环境变量或运行时输入)
    WRDS_USERNAME = None  # 设为None，将在运行时提示输入
    
    # Eikon App Key (通过环境变量或运行时输入)
    EIKON_APP_KEY = None  # 设为None，将在运行时提示输入
    
    @classmethod
    def get_tickers_with_benchmark(cls):
        """返回包含基准的所有股票代码列表"""
        return cls.TECH_TICKERS + [cls.BENCHMARK_TICKER]
    
    @classmethod
    def get_date_splits(cls):
        """返回数据集划分的日期元组"""
        return (cls.START_DATE, cls.TRAIN_END_DATE, 
                cls.VAL_END_DATE, cls.END_DATE)
    
    @classmethod
    def get_model_params(cls):
        """返回模型参数字典"""
        return {
            'd_model': cls.D_MODEL,
            'nhead': cls.NHEAD,
            'num_layers': cls.NUM_LAYERS,
            'dropout': cls.DROPOUT
        }
    
    @classmethod
    def get_training_params(cls):
        """返回训练参数字典"""
        return {
            'batch_size': cls.BATCH_SIZE,
            'num_epochs': cls.NUM_EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'early_stopping_patience': cls.EARLY_STOPPING_PATIENCE
        }
    
    @classmethod
    def get_backtest_params(cls):
        """返回回测参数字典"""
        return {
            'initial_capital': cls.INITIAL_CAPITAL,
            'top_n': cls.TOP_N,
            'rebalance_freq': cls.REBALANCE_FREQ
        }
    
    @classmethod
    def setup_directories(cls):
        """创建项目所需的所有目录"""
        import os
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR,
            cls.RESULTS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"确保目录存在: {directory}")