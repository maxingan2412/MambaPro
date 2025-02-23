import logging
import os
import sys
import os.path as osp
from datetime import datetime


def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        # 生成时间戳字符串（格式示例：07102024_153045 表示 2024年7月10日 15:30:45）
        # timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # 根据 if_train 选择前缀
        log_prefix = "train_log" if if_train else "test_log"

        # 构建带时间戳的文件名
        log_filename = f"{log_prefix}_{timestamp}.txt"
        log_path = osp.join(save_dir, log_filename)

        fh = logging.FileHandler(log_path, mode='w')  # 统一用变量 log_path
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
