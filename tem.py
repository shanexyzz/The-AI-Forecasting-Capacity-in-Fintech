import sys
print("--- 当前正在使用的 Python 路径 ---")
print(sys.executable)

try:
    import wrds
    print("\n✅ 成功！wrds 模块可以正常使用。")
except ImportError:
    print("\n❌ 失败！当前环境里依然没有 wrds。")