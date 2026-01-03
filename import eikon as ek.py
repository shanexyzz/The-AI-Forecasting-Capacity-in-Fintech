import eikon as ek
import pandas as pd

# 1. 设置 AppKey
APP_KEY = '7608acee9ff64ce69b595c436d47e087ad596dcd' # 请确保是在 Eikon APPKEY 里面生成的
ek.set_app_key(APP_KEY)


# 获取 腾讯 和 苹果 的 IBES EPS 平均预测、分析师人数、以及目标价
# 获取过去一段时间内，每日更新的 EPS 预测均值
df_history, err = ek.get_data(
    'AAPL.O', 
    ['TR.EPSMean(Period=FY1).calcdate', 'TR.EPSMean(Period=FY1)', 'TR.EPSMean(Period=FY2)',   # EPS 预测均值
        'TR.RecMean'],               # 分析师推荐均值],
    {'SDate': '2023-01-01', 'EDate': '2023-12-31', 'Frq': 'D'}
)
print(df_history)
