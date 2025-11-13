import pandas as pd
import requests
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

df = pd.read_csv("/home/pushpaka/ml_test/dataset/test.csv")
save_dir = "images/test"
os.makedirs(save_dir, exist_ok=True)

def download_image(args):
    sample_id, image_url = args
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        if response.status_code == 200:
            file_path = os.path.join(save_dir, f"{sample_id}.jpg")
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return f"Saved {sample_id}"
        else:
            return f"Failed {sample_id}, status code: {response.status_code}"
    except Exception as e:
        return f"Error {sample_id}: {e}"

if __name__ == "__main__":
    num_processes = 40
    args_list = [(row['sample_id'], row['image_link']) for _, row in df.iterrows()]
    with Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(download_image, args_list), total=len(args_list)):
            pass
