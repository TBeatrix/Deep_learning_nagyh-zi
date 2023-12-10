# Deep_Learning_Project

**Team Name: Pain & Panic**  
Team Members:  
- T63K63: Tugyi Beatrix  
- IT9P0Z: Heizer Levente  

**Project Description:** Link Prediction with Graph Neural Networks on the Facebook dataset

# Final
The files of the final submisson are in the Final directory. 
- **dl_final.ipynb** Ez a fájl tartalmazza az elkészült kódokat.
- **df_final_optimalization**

  
- **Final** directory: This is the version of the code, that can run in a container with docker. It contains a Dockerfile.
    Within the directory, we can start the docker and build the environment with the following code (image_name and tag can be anything):
    This also include the data acquisition step.
    
   **docker build . -t {image_name:tag}**
  
   After this we can run the code pipeline with the following command: 
  
  **docker run -it {image_name:tag}**
  This will run the data preprocessing, the training and the evaluation. In will log the result of the data preprocessing and traingin in the console.
  
  This is a good demonstration, that our code can be ran in a docker. However **the results of the hiperparameter optimalization, the visualizations and the traing for the whole graph are only avaible in the notebooks.**
---



## Milestone 2

how to run the pipeline: You can run the pipeline from the milestone2.ipynb

how to train and evaluate the models: At the end of the nodebooks there are to functions to train and evaluate.
It runs the model on all the Facebook graph separetly and give back the average test score.

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
  - https://arxiv.org/pdf/1611.07308.pdf

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
