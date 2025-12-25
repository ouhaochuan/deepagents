#%%
import dmPython
from langchain_core.tools import tool
from urllib.parse import urlparse
import urllib.parse
from pydantic import BaseModel, Field

def parse_connection_string(connection_string):
    """
    解析达梦数据库连接字符串，提取用户、密码、服务器、端口等信息
    支持格式如: dm://user:password@server:port?schema=schema_name
    """
    # 使用正则表达式解析连接字符串
    # 格式: protocol://user:password@server:port?parameters
    parsed = urlparse(connection_string)
    
    # 提取用户名和密码
    user = parsed.username if parsed.username else 'SYSDBA'
    password = parsed.password if parsed.password else 'SYSDBA001'
    
    # 提取服务器和端口
    server = parsed.hostname if parsed.hostname else '127.0.0.1'
    port = parsed.port if parsed.port else 30236

    # 默认模式
    # 从查询参数中提取 schema
    query_params = urllib.parse.parse_qs(parsed.query)
    schema = query_params.get('schema', ['SYSDBA'])[0]
    
    # 返回连接参数字典
    return {
        'user': user,
        'password': password,
        'server': server,
        'port': port,
        'schema': schema,
        'autoCommit': True
    }

def create_dm_database_tools(connection_string: str):
    """
    创建达梦数据库工具
    
    Args:
        connection_string: 数据库连接字符串，格式如: dm://user:password@server:port?schema=schema_name
    
    Returns:
        包含数据库工具的列表
    """

    class ListDatabaseToolInput(BaseModel):
        """输入参数为空，仅用于获取数据库中的表列表"""
        empty_input: str = Field(
            default="", 
            description="此参数为空字符串，仅作为占位符，因为工具需要至少一个参数"
        )
    @tool(args_schema=ListDatabaseToolInput)
    def dm_list_sql_database_tool(empty_input: str) -> str:
        """列出数据库中的所有表名。"""
        try:
            # 连接到数据库
            conn_params = parse_connection_string(connection_string)
            conn = dmPython.connect(**conn_params)
            cursor = conn.cursor()
            
            # 获取schema信息
            schema = conn_params.get('schema', 'SYSDBA')
            
            # 查询指定schema下的所有表名
            cursor.execute("SELECT TABLE_NAME FROM ALL_TABLES WHERE OWNER = UPPER(?) ORDER BY TABLE_NAME", (schema,))
            tables = [row[0] for row in cursor.fetchall()]
            
            result = ", ".join(tables)
            cursor.close()
            conn.close()
            
            return result if result else "数据库中没有表"
        except Exception as e:
            return f"错误: {str(e)}"


    class InfoDatabaseToolInput(BaseModel):
        """输入参数为表名，用于获取指定表的结构信息"""
        table_name: str = Field(
            ..., 
            description="需要查询结构的表名"
        )
    @tool(args_schema=InfoDatabaseToolInput)
    def dm_info_sql_database_tool(table_name: str) -> str:
        """获取指定表的结构信息。"""
        try:
            if not table_name:
                return "错误: 请输入表名"
                
            # 连接到数据库
            conn_params = parse_connection_string(connection_string)
            conn = dmPython.connect(**conn_params)
            cursor = conn.cursor()
            
            # 获取schema信息
            schema = conn_params.get('schema', 'SYSDBA')
            
            # 查询表结构信息
            sql = """
              SELECT 
                  utc.COLUMN_NAME, 
                  utc.DATA_TYPE, 
                  utc.DATA_LENGTH, 
                  utc.DATA_PRECISION, 
                  utc.DATA_SCALE, 
                  utc.NULLABLE, 
                  utc.DATA_DEFAULT AS COLUMN_DEFAULT, 
                  ucc.COMMENTS
              FROM ALL_TAB_COLUMNS utc
              LEFT JOIN ALL_COL_COMMENTS ucc 
                  ON utc.OWNER = ucc.OWNER 
                  AND utc.TABLE_NAME = ucc.TABLE_NAME 
                  AND utc.COLUMN_NAME = ucc.COLUMN_NAME
              WHERE utc.OWNER = UPPER(?)
                  AND utc.TABLE_NAME = UPPER(?)
              ORDER BY utc.COLUMN_ID
            """
            cursor.execute(sql, (schema, table_name))
            columns = cursor.fetchall()
            
            if not columns:
                return f"错误: 表 {table_name} 不存在或无法访问"
            
            # 构建表结构信息
            table_info = f"表 {table_name} 的结构信息:\n"
            table_info += "列名 | 数据类型 | 长度/精度 | 可空 | 默认值 | 备注\n"
            table_info += "-" * 80 + "\n"
            for col in columns:
                col_name, data_type, data_len, data_precision, data_scale, nullable, default_val, *comments = col
                # 根据数据类型构造长度/精度信息
                if data_type in ['VARCHAR', 'VARCHAR2', 'CHAR', 'NVARCHAR2', 'NCHAR']:
                    length_info = str(data_len)
                elif data_type in ['NUMBER', 'DECIMAL']:
                    if data_precision and data_scale:
                        length_info = f"{data_precision},{data_scale}"
                    elif data_precision:
                        length_info = str(data_precision)
                    else:
                        length_info = ""
                else:
                    length_info = ""
                    
                comment = comments[0] if comments and comments[0] else ""
                table_info += f"{col_name} | {data_type} | {length_info} | {nullable} | {default_val} | {comment}\n"
            
            cursor.close()
            conn.close()
            
            return table_info
        except Exception as e:
            return f"错误: {str(e)}"


    class QueryDatabaseToolInput(BaseModel):
        """输入参数为SQL查询语句，用于执行查询并返回结果"""
        query: str = Field(
            ..., 
            description="要执行的SQL查询语句，仅支持SELECT查询语句"
        )
    @tool(args_schema=QueryDatabaseToolInput)
    def dm_query_sql_database_tool(query: str) -> str:
        """执行SQL查询语句并返回结果。仅支持查询语句(SELECT)，不支持修改数据的语句。"""
        try:
            # 检查是否为SELECT语句，防止执行修改数据的语句
            query_upper = query.strip().upper()
            if not query_upper.startswith("SELECT"):
                if query_upper.startswith(("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "MERGE")):
                    return "错误: 此工具仅支持SELECT查询语句，不允许执行修改数据的语句。"
            
            # 连接到数据库
            conn_params = parse_connection_string(connection_string)
            conn = dmPython.connect(**conn_params)
            cursor = conn.cursor()
            
            # 执行查询
            cursor.execute(query)
            results = cursor.fetchall()
            
            # 获取列名
            column_names = [desc[0] for desc in cursor.description]
            
            # 构造结果
            if not results:
                result_str = "查询结果为空"
            else:
                # 限制返回结果数量以防止输出过长
                max_rows = 100
                results = results[:max_rows]
                
                result_str = "查询结果:\n"
                result_str += " | ".join(column_names) + "\n"
                result_str += "-" * (len(" | ".join(column_names)) + len(column_names) - 1) + "\n"
                for row in results:
                    result_str += " | ".join(str(cell) if cell is not None else "NULL" for cell in row) + "\n"
            
            if len(results) == max_rows:
                result_str += f"\n... 结果被限制为前{max_rows}行，请优化查询条件以获取更精确的结果。"
            
            cursor.close()
            conn.close()
            
            return result_str
        except Exception as e:
            return f"错误: {str(e)}"


    tools = [
        dm_info_sql_database_tool,      # 获取表结构信息
        dm_list_sql_database_tool,      # 列出表
        dm_query_sql_database_tool      # 执行查询
    ]
    
    return tools

def test_database_tools():
    """
    测试数据库工具创建方法
    注意：需要替换为实际的数据库连接字符串才能运行
    """
    # 示例连接字符串，需要替换为实际的数据库连接信息
    test_connection_string = "dm://SYSDBA:SYSDBA001@10.196.3.73:30236?schema=guinsoft_base_sppe"
    
    try:
        tools = create_dm_database_tools(test_connection_string)
        
        print(f"成功创建了 {len(tools)} 个数据库工具:")
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name}: {tool.description}")
        
        # 获取各个工具
        list_tool = tools[1]  # 列出表工具
        info_tool = tools[0]  # 获取表结构工具
        query_tool = tools[2]  # 查询工具
        
        # 测试列出表
        print("\n--- 测试列出表 ---")
        try:
            tables = list_tool.invoke({})
            print(f"数据库中的表: {tables}")
        except Exception as e:
            print(f"列出表时发生错误: {e}")
        
        # 测试获取schema（需要指定表名）
        print("\n--- 测试获取表结构 (使用第一个表) ---")
        try:
            if tables and tables != "数据库中没有表":
                # 假设存在表，获取第一个表的结构
                sample_table = tables.split(',')[0].strip() if ',' in tables else tables.strip()
                if sample_table:
                    schema_info = info_tool.invoke({"table_name": sample_table})
                    print(f"表 {sample_table} 的结构: {schema_info}")
        except Exception as e:
            print(f"获取表结构时发生错误: {e}")
        
        # 测试执行查询（示例查询）
        print("\n--- 测试执行查询 ---")
        try:
            first_table = tables.split(',')[0].strip() if ',' in tables else tables.strip()
            sample_query = f"SELECT * FROM {first_table} LIMIT 5"  # 查询第一个表的前5行数据
            query_result = query_tool.invoke({"query": sample_query})
            print(f"查询结果: {query_result}")
        except Exception as e:
            print(f"执行查询时发生错误: {e}")
            print("尝试使用另一个简单的查询示例:")
            try:
                simple_query = "SELECT SYSDATE as current_date from dual"  # 达梦获取当前日期
                query_result = query_tool.invoke({"query": simple_query})
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