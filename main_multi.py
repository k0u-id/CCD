import os
import traceback
import logging
from config import get_configs
from train import run_train
from test import run_test

def main():
    P = get_configs()
    print(P, '\n')
    print('###### Train start ######')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(P['save_path'], 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    run_train(P, logger)
    # run_test(P, logger)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
