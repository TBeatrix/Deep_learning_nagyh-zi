FROM python:3.9.12

RUN apt-get update &&  \
	apt install -y 

ENV HOME /home/PandP
RUN mkdir -p $HOME/project1
WORKDIR $HOME/project1

COPY . .

RUN pip install --upgrade pip
RUN pip install torch==1.11.0
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -rm -d $HOME -s /bin/bash -g root -G sudo -u 1000 PandP
RUN echo 'PandP:PandP' | chpasswd

SHELL ["/bin/bash", "-l", "-c"]

RUN python src/main.py 
ENTRYPOINT python src/data_preprocess.py 
