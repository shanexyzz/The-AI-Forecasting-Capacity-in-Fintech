import pandas as pd
import numpy as np

# ==========================================
# 1. 读入 V1(宽表) 和 V2(长表) 的数据
# ==========================================
# 请替换为你实际保存 V1 和 V2 数据的路径
try:
    # 假设你之前跑 V1 时存的文件叫 features.parquet
    v1_features = pd.read_parquet('./data_for_colab/features.parquet') 
    # 假设这是你刚刚跑出来的 V2 5天周期的长表
    v2_features = pd.read_parquet('./data_for_colab/features_v2_60d_5d.parquet') 
except Exception as e:
    print(f"读取文件失败，请检查路径: {e}")

# ==========================================
# 2. 设定抽查目标 (挑一个良辰吉日和一只股票)
# ==========================================
# 我们从 V2 里面随便挑第 100 天的日期
test_date = v2_features.index.get_level_values('date').unique()[100] 
test_ticker = 'AAPL' # 查苹果
test_feature = 'rsi' # 查相对强弱指标

print("\n" + "="*55)
print(f" 🕵️‍♂️ 数据对齐抽查: {test_date.date()} | {test_ticker} | 特征: {test_feature}")
print("="*55)

# ==========================================
# 3. 提取对比数值
# ==========================================
# 在 V1 宽表中，列名是 AAPL_rsi
v1_col_name = f"{test_ticker}_{test_feature}"

if v1_col_name in v1_features.columns and (test_date in v1_features.index):
    # A. 提取 V1 的值
    v1_value = v1_features.loc[test_date, v1_col_name]
    print(f"👉 [V1 原始宽表] (仅包含过去60天时序 Z-Score) : {v1_value:.6f}")
    
    # B. 提取 V2 的值 (注意长表是双重索引)
    try:
        v2_value = v2_features.loc[(test_date, test_ticker), test_feature]
        print(f"👉 [V2 面板长表] (加入了全市场截面 Z-Score)  : {v2_value:.6f}")
    except KeyError:
        print("V2 中没有找到该股票当天的数据 (可能被清洗掉了)。")
        
    # ==========================================
    # 4. 手动硬算，验证逻辑！
    # ==========================================
    print("\n--- 🧠 正在手动复现 V2 的截面数学转换 ---")
    
    # 找出 V1 宽表中，在 test_date 这一天，全市场所有股票的 RSI 值
    all_rsi_cols = [c for c in v1_features.columns if c.endswith(f"_{test_feature}")]
    v1_cross_section = v1_features.loc[test_date, all_rsi_cols]
    
    # # 【见证奇迹】：把等于 0 的假数据（未上市/停牌）全部扔掉！
    # v1_cross_section_clean = v1_cross_section[v1_cross_section != 0]

    # # 用真正活着的股票算均值和标准差
    # cs_mean_clean = v1_cross_section_clean.mean()
    # cs_std_clean = v1_cross_section_clean.std()

    # manual_v2_value_clipped = np.clip((v1_value - cs_mean_clean) / cs_std_clean, -3.0, 3.0)
    
    # print(f"清洗掉0值后的真实均值: {cs_mean_clean:.6f}")
    # print(f"清洗掉0值后的真实标准差: {cs_std_clean:.6f}")
    # print(f"最终修正后的终极特征值: {manual_v2_value_clipped:.6f}")

    # 计算全市场这一天的 RSI 均值和标准差
    cs_mean = v1_cross_section.mean()
    cs_std = v1_cross_section.std()
    
    # 拿 AAPL 的 V1 值，减去均值，除以标准差 (这就是你极速版代码做的事)
    manual_v2_value = (v1_value - cs_mean) / cs_std
    
    # 模拟 V2 中的极值裁剪 clip(-3, 3)
    manual_v2_value_clipped = np.clip(manual_v2_value, -3.0, 3.0)
    
    print(f"全市场当日 {test_feature} 均值: {cs_mean:.6f}")
    print(f"全市场当日 {test_feature} 标准差: {cs_std:.6f}")
    print(f"手动计算得出的终极特征值: {manual_v2_value_clipped:.6f}")
    print("-" * 55)
    
    # 对比差距 (考虑到 float32 和 float64 的精度差异，允许 1e-4 的误差)
    if np.isclose(v2_value, manual_v2_value_clipped, atol=1e-4):
        print("✅ 验证成功！完美吻合！")
        print("结论：你写的极速版截面标准化逻辑严丝合缝，没有任何数据错位！")
    else:
        print("❌ 有差异！请检查是否是某一步填空 (fillna) 或剔除 (dropna) 导致的基数不同。")
else:
    print(f"在 V1 中没有找到 {v1_col_name}，或者当天没数据，请换个 ticker 或 date 试试。")