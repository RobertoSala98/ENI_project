FROM python:3.9.6
ENV MY_DIR=/ENI_project
WORKDIR ${MY_DIR}
COPY . .
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/aMLLibrary/aMLLibrary.git --recurse-submodules

CMD bash