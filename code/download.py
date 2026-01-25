import requests
import os

# Define a function to download a file from a URL
def download_file(url, folder):
    response = requests.get(url)
    if response.status_code == 200:
        # Extract the filename from the URL
        filename = os.path.join(folder, url.split("/")[-1])
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url}")

# Specify the path to the file containing links
links_file = r"E:\my works\andaman\data\nc\prov.data_fetch+dNOBM_MON_R2017_h+t20140101000000_20201231235959.txt"

# Create a folder to save the downloaded files (if it doesn't exist)
output_folder = r"E:\my works\andaman\data\nc\mld"
os.makedirs(output_folder, exist_ok=True)

# Read links from the file and download each one
with open(links_file, 'r') as file:
    links = file.readlines()
    for link in links:
        link = link.strip()  # Remove leading/trailing whitespace and newline characters
        download_file(link, output_folder)
