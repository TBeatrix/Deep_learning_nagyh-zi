# Deep_learning_nagyhazi

**Team name: Pain & Panic**
team members: T63K63 - Tugyi Beatrix
              IT9P0Z - Heizer Levente

Project description: Friend Recommendation with Graph Neural Networks on the Facebook and Twitter datasets.

## Milestone 1
functions of the files in the repository:

milestone1.ipynb: The whole code in Goggle Colab file. It contains extra visualizations and the Facebook and Twitter graph.
In the end we will hava a Graph object with all of the nessesery properties and the test/train/val masks.

Milestone1        Directory to run in docker

  - src/main.py   Script for data downloading
  - src/data_preprocess  Script for data preprocessing and splitting
  - Dockerfile    
  - requirements.txt packages and their versions for the Dockerfile

related works: 

datasets:  https://snap.stanford.edu/data/ego-Facebook.html, https://snap.stanford.edu/data/ego-Twitter.html

J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego Networks. NIPS, 2012.
           
code:
Node2Vec node preprocessing: https://github.com/aditya-grover/node2vec, 
           
https://github.com/VHRanger/nodevectors/blob/master/nodevectors/node2vec.py
           
https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html

**how to run it:**

Run the ipynb notebook in Google Colab

OR

Clone this repository

Go to the Milestone1 directory in your terminal

  docker build . -t {image_name:tag}
  
docker run -it {image_name:tag}


  
  
