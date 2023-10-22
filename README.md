# Deep_Learning_Project

**Team Name: Pain & Panic**  
Team Members:  
- T63K63: Tugyi Beatrix  
- IT9P0Z: Heizer Levente  

**Project Description:** Friend Recommendation using Graph Neural Networks on Facebook and Twitter Datasets

## Milestone 1
Description of the repository contents:

- `milestone1.ipynb`: This file contains the entire code, housed in a Google Colab notebook. It encompasses visualizations of the Facebook and Twitter graphs, a graph object with all requisite properties, and the test/train/validation masks.

- `Milestone1` Directory (for Docker execution):  
  - `src/main.py`: Script for data downloading  
  - `src/data_preprocess`: Script for data preprocessing and splitting  
  - `Dockerfile`: Contains the commands a user could call on the command line to assemble an image  
  - `requirements.txt`: Lists packages and their versions required for the Dockerfile

**Related Works:**  

- **Datasets:**  
  - [Facebook Dataset](https://snap.stanford.edu/data/ego-Facebook.html)  
  - [Twitter Dataset](https://snap.stanford.edu/data/ego-Twitter.html)

- **References:**  
  - J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego Networks. NIPS, 2012.

- **Code:**  
  - Node2Vec Node Preprocessing:
    - [GitHub - aditya-grover/node2vec](https://github.com/aditya-grover/node2vec)  
    - [GitHub - VHRanger/nodevectors](https://github.com/VHRanger/nodevectors/blob/master/nodevectors/node2vec.py)  
    - [NetworkX Documentation](https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html)

**Execution Instructions:**

1. Execute the `milestone1.ipynb` notebook in Google Colab.

   **- OR -**

2. Clone this repository, navigate to the `Milestone1` directory in your terminal, and run the following commands:  

```shell
docker build . -t {image_name:tag}
docker run -it {image_name:tag}
```
