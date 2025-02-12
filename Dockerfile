FROM python:3.9.6

RUN git clone https://github.com/aMLLibrary/aMLLibrary.git --recurse-submodules

ENV MY_DIR=/ENI_project
WORKDIR ${MY_DIR}

COPY functions.py functions.py
COPY main.py main.py
COPY utils.py utils.py
COPY config config
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["python", "main.py"]