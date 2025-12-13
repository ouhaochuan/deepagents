"""Custom tools for the CLI agent."""

from typing import Any, Literal

import requests
from markdownify import markdownify
from tavily import TavilyClient

from deepagents_cli.config import settings

# Initialize Tavily client if API key is available
tavily_client = TavilyClient(api_key=settings.tavily_api_key) if settings.has_tavily else None


def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: str | dict | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make HTTP requests to APIs and web services.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        headers: HTTP headers to include
        data: Request body data (string or dict)
        params: URL query parameters
        timeout: Request timeout in seconds

    Returns:
        Dictionary with response data including status, headers, and content
    """
    try:
        kwargs = {"url": url, "method": method.upper(), "timeout": timeout}

        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        if data:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data

        response = requests.request(**kwargs)

        try:
            content = response.json()
        except:
            content = response.text

        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": content,
            "url": response.url,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request timed out after {timeout} seconds",
            "url": url,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Request error: {e!s}",
            "url": url,
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "content": f"Error making request: {e!s}",
            "url": url,
        }


def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Search the web using Tavily for current information and documentation.

    This tool searches the web and returns relevant results. After receiving results,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: Search topic type - "general" for most queries, "news" for current events
        include_raw_content: Include full page content (warning: uses more tokens)

    Returns:
        Dictionary containing:
        - results: List of search results, each with:
            - title: Page title
            - url: Page URL
            - content: Relevant excerpt from the page
            - score: Relevance score (0-1)
        - query: The original search query

    IMPORTANT: After using this tool:
    1. Read through the 'content' field of each result
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. Cite sources by mentioning the page titles or URLs
    5. NEVER show the raw JSON to the user - always provide a formatted response
    """
    if tavily_client is None:
        return {
            "error": "Tavily API key not configured. Please set TAVILY_API_KEY environment variable.",
            "query": query,
        }

    try:
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except Exception as e:
        return {"error": f"Web search error: {e!s}", "query": query}


def fetch_url(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch content from a URL and convert HTML to markdown format.

    This tool fetches web page content and converts it to clean markdown text,
    making it easy to read and process HTML content. After receiving the markdown,
    you MUST synthesize the information into a natural, helpful response for the user.

    Args:
        url: The URL to fetch (must be a valid HTTP/HTTPS URL)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - url: The final URL after redirects
        - markdown_content: The page content converted to markdown
        - status_code: HTTP status code
        - content_length: Length of the markdown content in characters

    IMPORTANT: After using this tool:
    1. Read through the markdown content
    2. Extract relevant information that answers the user's question
    3. Synthesize this into a clear, natural language response
    4. NEVER show the raw markdown to the user unless specifically requested
    """
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DeepAgents/1.0)"},
        )
        response.raise_for_status()

        # Convert HTML content to markdown
        markdown_content = markdownify(response.text)

        return {
            "url": str(response.url),
            "markdown_content": markdown_content,
            "status_code": response.status_code,
            "content_length": len(markdown_content),
        }
    except Exception as e:
        return {"error": f"Fetch URL error: {e!s}", "url": url}

##%%
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    UnstructuredExcelLoader,
)
import os
from typing import Optional

def load_word(file_path: str, element_mode: Optional[bool] = None) -> str:
    """Load a word from a file.

    This tool loads a word from a file and returns it's content as json format.

    Args:
        file_path: The absolute path to the file containing the word
        element_mode: Whether to load the document as elements or not

    Returns:
        The word file content as json format loaded from the file
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return '{"error": "File not found"}'

    if element_mode:
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
    else:
        loader = UnstructuredWordDocumentLoader(file_path)
    data = loader.load()
    json_text = serialize_data_to_json(data)
    return json_text

def load_excel(file_path: str, element_mode: Optional[bool] = None) -> str:
    """Load an Excel file.

    This tool loads an Excel file and returns it's content as json format.

    Args:
        file_path: The absolute path to the file containing the Excel file
        element_mode: Whether to load the document as elements or not

    Returns:
        The Excel file content as json format loaded from the file
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return '{"error": "File not found"}'
    
    if element_mode:
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    else:
        loader = UnstructuredExcelLoader(file_path)

    data = loader.load()
    json_text = serialize_data_to_json(data)
    return json_text

def load_pdf(file_path: str, element_mode: Optional[bool] = None) -> str:
    """Load a PDF from a file.

    This tool loads a PDF from a file and returns it's content as json format.

    Args:
        file_path: The absolute path to the file containing the PDF
        element_mode: Whether to load the document as elements or not

    Returns:
        The PDF content as json format loaded from the file
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return '{"error": "File not found"}'
    
    if element_mode:
        loader = UnstructuredPDFLoader(file_path, mode="elements")
    else:
        loader = UnstructuredPDFLoader(file_path)

    data = loader.load()
    json_text = serialize_data_to_json(data)
    return json_text

import json
def serialize_data_to_json(data):
    """
    将加载的数据序列化为JSON文本
    
    Args:
        data: 从UnstructuredWordDocumentLoader.load()返回的数据
        
    Returns:
        str: 序列化后的JSON文本
    """
    # 处理Document对象，提取可序列化的属性
    serializable_data = []
    for doc in data:
        serializable_doc = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        serializable_data.append(serializable_doc)
    
    # 转换为JSON字符串
    return json.dumps(serializable_data, ensure_ascii=False, indent=2)

# #%%
# output = load_word("c:\\Users\\ouhaochuan\\Downloads\\20250928073821.docx")
# print(output)

# #%%
# output = load_excel("C:\\Users\\ouhaochuan\\Downloads\\工日提交.xlsx")
# print(output)

# #%%
# output = load_pdf("C:\\Users\\ouhaochuan\\Documents\\WeChat Files\\ouhaochuan\\FileStorage\\File\\2025-05\\110kV塘下（先锋）至燕罗双回线路工程（电缆部分） 441-S6967Z-D0401 (4)\\17 主要设备材料清册.pdf",
#                   True)
# print(output)

def copy_and_rename_template_file(template_file_path: str, new_file_path: str) -> str:
    """Copy a template file and rename it.

    This tool copies a template file and renames it.

    Args:
        template_file_path: The absolute path to the template file
        new_file_path: The absolute path to the new file

    Returns:
        The absolute path to the new file
    """
    import shutil
    import os
    
    # Check if template file exists
    if not os.path.exists(template_file_path):
        raise FileNotFoundError(f"Template file not found: {template_file_path}")
    
    # Create directory for new file if it doesn't exist
    new_file_dir = os.path.dirname(new_file_path)
    if new_file_dir and not os.path.exists(new_file_dir):
        os.makedirs(new_file_dir)
    
    # Copy and rename the file
    shutil.copy2(template_file_path, new_file_path)
    
    # Return the absolute path to the new file
    return os.path.abspath(new_file_path)
