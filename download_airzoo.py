import time 
from huggingface_hub import snapshot_download, login
from tenacity import retry, wait_exponential, before_sleep

login("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")

# def if_retry(retry_state):
#     print("\nWas retry\n")

# @retry(wait=wait_exponential(multiplier=1, min=2, max=10), before_sleep=if_retry)
# def download():
#     import os
#     print("\nSET VARIABLE\n")
#     os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

#     print("\nSTART DOWNLOAD\n")
#     snapshot_download(
#     repo_id="RingoWRW97/AirZoo-Real", 
#     repo_type="dataset",  
#     local_dir=r"C:\Users\t.asmus\Desktop\data\AirZoo" 
#     )   



def download():
    from datasets import load_dataset
    print("\nSTART DOWNLOAD\n")
    # load_dataset("RingoWRW97/AirZoo-Real", cache_dir=r"C:\Users\t.asmus\Desktop\data\AirZoo")
    load_dataset("haiduonghuynhle/AnyVisLoc", cache_dir=r"C:\Users\t.asmus\Desktop\data\AnyVisLoc")
    print("\nDONE\n")

download()