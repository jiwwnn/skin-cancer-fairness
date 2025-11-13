import pandas as pd 
import os
import urllib.request

save_folder = "data/fitzpatrick17_images"
os.makedirs(save_folder, exist_ok=True)

# 'non-neoplastic' label 제외
df = pd.read_csv('/dshome/ddualab/jiwon/skin_cancer_fairness/data/fitzpatrick17k.csv')

# User-Agent 설정 (HTTP Error 406 방지)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
}

def download_image(image_url, file_name):
    try:
        req = urllib.request.Request(image_url, headers=headers)

        with urllib.request.urlopen(req) as response, open(file_name, 'wb') as out_file:
            out_file.write(response.read())
        
        print(f"Downloaded: {file_name}")
    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code} for {image_url}")
    except urllib.error.URLError as e:
        print(f"URLError: {e.reason} for {image_url}")
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")

for _, row in df.iterrows():
    url = row['url']
    file_name = os.path.join(save_folder, row['md5hash'] + '.jpg')

    if isinstance(url, str) and url.startswith("http"):
        download_image(url, file_name)
    else:
        print(f"Skipping invalid URL: {url}")

