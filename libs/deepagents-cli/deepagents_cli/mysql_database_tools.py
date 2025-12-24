#%%
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDatabaseTool,
)

def create_mysql_database_tools(connection_string):
    """
    创建数据库工具
    
    Args:
        connection_string: 数据库连接字符串
    
    Returns:
        包含数据库工具的列表
    """
    db = SQLDatabase.from_uri(connection_string, lazy_table_reflection=True)
    
    # 只创建3个基本工具，无LLM依赖
    tools = [
        InfoSQLDatabaseTool(db=db, name="mysql_info_sql_database_tool"),      # 列出表
        ListSQLDatabaseTool(db=db, name="mysql_list_sql_database_tool"),      # 获取schema  
        QuerySQLDatabaseTool(db=db, name="mysql_query_sql_database_tool")     # 执行查询
    ]
    
    return tools

# # SQL Server连接字符串格式
# # 替换为你的实际连接信息
# connection_string = "mssql+pyodbc://username:password@server:1433/database?driver=ODBC+Driver+17+for+SQL+Server"
# 
# db = SQLDatabase.from_uri(connection_string)
# 
# # 只创建3个基本工具，无LLM依赖
# tools = [
#     InfoSQLDatabaseTool(db=db),      # 列出表
#     ListSQLDatabaseTool(db=db),      # 获取schema  
#     QuerySQLDataBaseTool(db=db)      # 执行查询
# ]

def test_database_tools():
    """
    测试数据库工具创建方法
    注意：需要替换为实际的数据库连接字符串才能运行
    """
    # 示例连接字符串，需要替换为实际的数据库连接信息
    test_connection_string = "mysql+pymysql://root:Guinsoft%403306@10.196.85.186:3306/guinsoft_base_zy"
    
    try:
        tools = create_mysql_database_tools(test_connection_string)
        
        print(f"成功创建了 {len(tools)} 个数据库工具:")
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name}: {tool.description}")
        
        # 获取各个工具
        info_tool = tools[0]
        list_tool = tools[1]
        query_tool = tools[2]
        
        # 测试列出表
        print("\n--- 测试列出表 ---")
        try:
            tables = list_tool.run("")
            print(f"数据库中的表: {tables}")
        except Exception as e:
            print(f"列出表时发生错误: {e}")
        
        # 测试获取schema（需要指定表名）
        print("\n--- 测试获取表结构 (使用第一个表) ---")
        try:
            if tables:
                # 假设存在表，获取第一个表的结构
                sample_table = tables.split(',')[0].strip() if ',' in tables else tables.strip()
                schema_info = info_tool.run(sample_table)
                print(f"表 {sample_table} 的结构: {schema_info}")
        except Exception as e:
            print(f"获取表结构时发生错误: {e}")
        
        # 测试执行查询（示例查询）
        print("\n--- 测试执行查询 ---")
        try:
            sample_query = "SELECT * FROM " + (tables.split(',')[0].strip() if tables and ',' in tables else tables.strip() if tables else "your_table_name") + " LIMIT 5"
            query_result = query_tool.run(sample_query)
            print(f"查询结果: {query_result}")
        except Exception as e:
            print(f"执行查询时发生错误: {e}")
            print("尝试使用一个简单的查询示例:")
            try:
                simple_query = "SELECT 1 as test"  # 简单查询测试
                query_result = query_tool.run(simple_query)
                print(f"简单查询结果: {query_result}")
            except Exception as se:
                print(f"简单查询也失败: {se}")
        
        return tools
        
    except Exception as e:
        print(f"创建数据库工具时发生错误: {e}")
        print("请确保连接字符串正确，并且数据库服务可用")
        return None

#%%
if __name__ == "__main__":
    test_database_tools()