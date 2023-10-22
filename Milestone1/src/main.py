import re

import subprocess

def get_ego_indexes(files):
  pattern = re.compile(r'\d+')
  ego_indexes = [int(pattern.search(s).group()) for s in files if pattern.search(s)]
  ego_indexes = sorted(set(ego_indexes))
  return ego_indexes

if __name__ == '__main__':
  url = "https://snap.stanford.edu/data/facebook.tar.gz"
  try:
    result = subprocess.run(["wget", url], check=True)
    if result.returncode == 0:
      print(f"File downloaded successfully from {url}")
    else:
      print(f"Error occurred while downloading from {url}")
  except subprocess.CalledProcessError:
      print(f"Error occurred while downloading from {url}")
  except FileNotFoundError:
      print("wget command not found. Please ensure wget is installed on your system.")
  
 
  try:
    result_files = subprocess.run(["tar", "xvzf", "facebook.tar.gz"],  capture_output=True, text=True, check=True) 
    if result_files.returncode == 0:
        print("Files extracted successfully:")
        print(result_files.stdout)
    else:
        print("Error occurred while extracting the archive.")
  except subprocess.CalledProcessError:
    print("Error occurred while extracting the archive.")
  except FileNotFoundError:
    print("tar command not found. Please ensure tar is installed on your system.")
#  files = ! tar xvzf facebook.tar.gz

  files = result_files.stdout.splitlines()
  Fb_ego_indexes = get_ego_indexes(files)
  with open('facebook/indexes.txt', 'w') as f:
    for index in Fb_ego_indexes:
        f.write(f"{index}\n")
