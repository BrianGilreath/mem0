import os
import unittest
from unittest.mock import patch, mock_open
from app.utils.memory import _get_docker_host_url, _fix_localhost_urls


class TestMemoryUtils(unittest.TestCase):

    @patch('os.path.exists')
    def test_get_docker_host_url_not_in_docker(self, mock_exists):
        mock_exists.return_value = False
        result = _get_docker_host_url()
        self.assertEqual(result, "localhost")

    @patch.dict(os.environ, {'OLLAMA_HOST': 'http://custom-host:11434'})
    @patch('os.path.exists')
    def test_get_docker_host_url_with_ollama_host(self, mock_exists):
        mock_exists.return_value = True
        result = _get_docker_host_url()
        self.assertEqual(result, "custom-host")

    @patch('socket.gethostbyname')
    @patch('os.path.exists')
    def test_get_docker_host_url_with_host_docker_internal(self, mock_exists, mock_gethostbyname):
        mock_exists.return_value = True
        mock_gethostbyname.return_value = "192.168.65.254"
        result = _get_docker_host_url()
        self.assertEqual(result, "host.docker.internal")

    def test_fix_localhost_urls_ollama_provider(self):
        config = {"provider": "ollama", "config": {}}
        with patch('app.utils.memory._get_docker_host_url', return_value='host.docker.internal'):
            result = _fix_localhost_urls(config)
        self.assertEqual(result["config"]["ollama_base_url"], "http://host.docker.internal:11434")

    def test_fix_localhost_urls_lmstudio_localhost(self):
        config = {
            "provider": "lmstudio",
            "config": {"lmstudio_base_url": "http://localhost:1234"}
        }
        with patch('app.utils.memory._get_docker_host_url', return_value='mock-value'):
            result = _fix_localhost_urls(config)
        self.assertEqual(result["config"]["lmstudio_base_url"], "http://mock-value:1234")

    def test_fix_localhost_urls_no_change_needed(self):
        config = {
            "provider": "openai",
            "config": {"base_url": "https://api.openai.com/v1"}
        }
        result = _fix_localhost_urls(config)
        self.assertEqual(result["config"]["base_url"], "https://api.openai.com/v1")


if __name__ == '__main__':
    unittest.main()