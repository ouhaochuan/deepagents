import unittest
from unittest.mock import patch, mock_open
import os
from pathlib import Path
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import EditResult

class TestFilesystemBackendEdit(unittest.TestCase):
    
    def setUp(self):
        self.backend = FilesystemBackend()
        
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('os.open')
    @patch('os.fdopen')
    def test_edit_success_replace_single_occurrence(self, mock_fdopen, mock_os_open, mock_is_file, mock_exists):
        # 模拟文件存在
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        # 模拟文件存在且包含要替换的字符串
        mock_file = mock_open(read_data="Hello world\n{\n这是一个测试}\nGoodbye world")
        mock_os_open.return_value = 123
        mock_fdopen.side_effect = [
            mock_file.return_value,  # 读取文件时
            mock_file.return_value   # 写入文件时
        ]
        
        result = self.backend.edit("test.txt", "{\n这是一个测试}", "你好", False)
        
        # 验证结果
        self.assertIsInstance(result, EditResult)
        self.assertEqual(result.path, "test.txt")
        self.assertEqual(result.occurrences, 1)
        self.assertIsNone(result.error)
        
        # 验证文件写入操作
        mock_file().write.assert_called_once_with("Hello world\n你好\nGoodbye world")
        
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('os.open')
    @patch('os.fdopen')
    def test_edit_success_replace_multiple_occurrences_with_replace_all(self, mock_fdopen, mock_os_open, mock_is_file, mock_exists):
        # 模拟文件存在
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        # 模拟文件存在且包含多个要替换的字符串
        mock_file = mock_open(read_data="Hello world\nThis is a Hello test\nGoodbye Hello")
        mock_os_open.return_value = 123
        mock_fdopen.side_effect = [
            mock_file.return_value,  # 读取文件时
            mock_file.return_value   # 写入文件时
        ]
        
        result = self.backend.edit("test.txt", "Hello", "Hi", True)
        
        # 验证结果
        self.assertIsInstance(result, EditResult)
        self.assertEqual(result.path, "test.txt")
        self.assertEqual(result.occurrences, 3)
        self.assertIsNone(result.error)
        
        # 验证文件写入操作
        mock_file().write.assert_called_once_with("Hi world\nThis is a Hi test\nGoodbye Hi")
        
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('os.open')
    @patch('os.fdopen')
    def test_edit_fail_multiple_occurrences_without_replace_all(self, mock_fdopen, mock_os_open, mock_is_file, mock_exists):
        # 模拟文件存在
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        # 模拟文件存在且包含多个要替换的字符串但未设置replace_all
        mock_file = mock_open(read_data="Hello world\nThis is a Hello test\nGoodbye Hello")
        mock_os_open.return_value = 123
        mock_fdopen.return_value = mock_file.return_value
        
        result = self.backend.edit("test.txt", "Hello", "Hi", False)
        
        # 验证结果
        self.assertIsInstance(result, EditResult)
        self.assertIsNotNone(result.error)
        self.assertIn("appears 3 times", result.error)
        
    @patch('pathlib.Path.exists')
    def test_edit_fail_file_not_found(self, mock_exists):
        # 模拟文件不存在
        mock_exists.return_value = False
        
        result = self.backend.edit("nonexistent.txt", "test", "replacement", False)
        
        # 验证结果
        self.assertIsInstance(result, EditResult)
        self.assertIsNotNone(result.error)
        self.assertIn("not found", result.error)
        
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('os.open')
    @patch('os.fdopen')
    def test_edit_fail_string_not_found(self, mock_fdopen, mock_os_open, mock_is_file, mock_exists):
        # 模拟文件存在
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        # 模拟文件存在但不包含要替换的字符串
        mock_file = mock_open(read_data="Hello world\nThis is a test\nGoodbye world")
        mock_os_open.return_value = 123
        mock_fdopen.return_value = mock_file.return_value
        
        result = self.backend.edit("test.txt", "Nonexistent", "Replacement", False)
        
        # 验证结果
        self.assertIsInstance(result, EditResult)
        self.assertIsNotNone(result.error)
        self.assertIn("String not found", result.error)

if __name__ == '__main__':
    unittest.main()