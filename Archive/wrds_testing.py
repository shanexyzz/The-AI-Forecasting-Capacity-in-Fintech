import wrds
db = wrds.Connection(wrds_username="xyzshane")  # 这里可以传入你的WRDS用户名
print("WRDS连接成功!")

#how to check which library I have access to
libraries = db.list_libraries()
print("可访问的库列表:")
print(libraries)

#check tables in crsp library
lib = db.list_tables(library='crsp')
print(lib)
