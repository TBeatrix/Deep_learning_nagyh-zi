# Deep_learning_nagyhazi

**Team name: Pain & Panic**
team members: T63K63 - Tugyi Beatrix
              IT9P0Z - Heizer Levente

Project description: Friend Recommendation with Graph Neural Networks on the Facebook and Twitter datasets.

##Milestone 1
functions of the files in the repository:
milestone1.ipynb:  A teljes kód google colabon futtatható változata.
Milestone1        Mappa saját gépen való futtatáshoz
  - src/main.py   A futtatandó kód
  - Dockerfile    Dockerban való futtatást előkészítő file
  - requirements.txt A Docker futtatásához szükséges csomagok és azok használt verziójának a felsorolása

related works: 
datasets:  https://snap.stanford.edu/data/ego-Facebook.html, https://snap.stanford.edu/data/ego-Twitter.html
           J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego Networks. NIPS, 2012.
code:
           Node2Vec node preprocessing: https://github.com/aditya-grover/node2vec,         
           https://github.com/VHRanger/nodevectors/blob/master/nodevectors/node2vec.py
           https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html

and how to run it:
Run the ipynb notebook in Google Colab

OR

Clone this repository
Go to the Milestone1 directory in your terminal
  docker build . -t {image_name:tag}
docker run -it {image_name:tag} bash
python3 src/data_preprocess.py
  
  
