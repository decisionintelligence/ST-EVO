import logging
import sys

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 输出到控制台
    ]
)

# 创建 logger 实例
logger = logging.getLogger("Reader")
