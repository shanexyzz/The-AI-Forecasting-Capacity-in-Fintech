# test_fields.py - 测试每个字段
import eikon as ek
import pandas as pd

# 设置你的Eikon App Key
ek.set_app_key('7608acee9ff64ce69b595c436d47e087ad596dcd')

def test_individual_fields():
    symbol = 'AAPL.O'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    fields = [
        ('TR.EPSMean(Period=FY1)', 'EPS FY1'),
        ('TR.EPSMean(Period=FY2)', 'EPS FY2'),
        ('TR.RecMean', 'Rec Mean')  # 备用
    ]
    
    for field_code, field_name in fields:
        print(f"\n{'='*60}")
        print(f"测试字段: {field_name} ({field_code})")
        print('='*60)
        
        try:
            # 获取数据
            result = ek.get_data(
                symbol,
                [f'{field_code}.calcdate', field_code],
                {'SDate': start_date, 'EDate': end_date, 'Frq': 'D'}
            )
            
            if isinstance(result, tuple):
                df, err = result
                if err:
                    print(f"错误: {err}")
                    continue
            else:
                df = result
            
            print(f"数据形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            
            if df is not None and not df.empty:
                print(f"\n前3行数据:")
                print(df.head(3))
                
                # 检查数据类型
                print(f"\n数据类型:")
                print(df.dtypes)
                
                # 检查日期列
                date_cols = [col for col in df.columns if 'Date' in str(col) or 'date' in str(col).lower()]
                print(f"可能的日期列: {date_cols}")
                
                if date_cols:
                    print(f"日期范围: {df[date_cols[0]].min()} 到 {df[date_cols[0]].max()}")
            
        except Exception as e:
            print(f"异常: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    print("开始测试各个字段...")
    test_individual_fields()