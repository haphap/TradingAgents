import os
os.environ.setdefault("PYTHONUTF8", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
