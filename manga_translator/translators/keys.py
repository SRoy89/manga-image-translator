import os
from dotenv import load_dotenv
load_dotenv()

# baidu
BAIDU_APP_ID = os.getenv('BAIDU_APP_ID', '') #ä½ çš„appid
BAIDU_SECRET_KEY = os.getenv('BAIDU_SECRET_KEY', '') #ä½ çš„å¯†é’¥
# youdao
YOUDAO_APP_KEY = os.getenv('YOUDAO_APP_KEY', '') # åº”ç”¨ID
YOUDAO_SECRET_KEY = os.getenv('YOUDAO_SECRET_KEY', '') # åº”ç”¨ç§˜é’¥
# deepl
DEEPL_AUTH_KEY = os.getenv('DEEPL_AUTH_KEY', '') #YOUR_AUTH_KEY
# openai
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5-chat-latest')

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768')

OPENAI_HTTP_PROXY = os.getenv('OPENAI_HTTP_PROXY') # TODO: Replace with --proxy
OPENAI_GLOSSARY_PATH = os.getenv('OPENAI_GLOSSARY_PATH', './dict/mit_glossary.txt') # OpenAIæœ¯è¯­è¡¨è·¯å¾„
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1') #ä½¿ç”¨api-for-open-llmä¾‹å­� http://127.0.0.1:8000/v1

# sakura
SAKURA_API_BASE = os.getenv('SAKURA_API_BASE', 'http://127.0.0.1:8080/v1') #SAKURA APIåœ°å�€
SAKURA_VERSION = os.getenv('SAKURA_VERSION', '0.9') #SAKURA APIç‰ˆæœ¬ï¼Œå�¯é€‰å€¼ï¼š0.9ã€�0.10ï¼Œé€‰æ‹©0.10åˆ™ä¼šåŠ è½½æœ¯è¯­è¡¨ã€‚
SAKURA_DICT_PATH = os.getenv('SAKURA_DICT_PATH', './dict/sakura_dict.txt') #SAKURA æœ¯è¯­è¡¨è·¯å¾„


CAIYUN_TOKEN = os.getenv('CAIYUN_TOKEN', '') # å½©äº‘å°�è¯‘APIè®¿é—®ä»¤ç‰Œ

# Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-002')

# deepseek
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_API_BASE  = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
DEEPSEEK_MODEL  = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat') # Or: "deepseek-reasoner"

# Together AI
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY', '')
TOGETHER_VL_MODEL = os.getenv('TOGETHER_VL_MODEL', 'Qwen/Qwen2.5-VL-72B-Instruct')

# ollama, with OpenAI API compatibility
CUSTOM_OPENAI_API_KEY = os.getenv('CUSTOM_OPENAI_API_KEY', 'ollama') # Unsed for ollama, but maybe useful for other LLM tools.
CUSTOM_OPENAI_API_BASE = os.getenv('CUSTOM_OPENAI_API_BASE', 'http://localhost:11434/v1') # Use OLLAMA_HOST env to change binding IP and Port.
CUSTOM_OPENAI_MODEL = os.getenv('CUSTOM_OPENAI_MODEL', '') # e.g "qwen2.5:7b". Make sure to pull and run it before use.
CUSTOM_OPENAI_MODEL_CONF = os.getenv('CUSTOM_OPENAI_MODEL_CONF', '') # e.g "qwen2".
