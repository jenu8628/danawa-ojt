FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install python3 -y
RUN apt-get install wget -y
# RUN apt-get install python-pip -y
# RUN apt-get iputils-ping -y
# RUN apt-get install python-mysqldb -y
# RUN apt-get install python-pymysql -y

RUN wget https://github.com/coder/code-server/releases/download/v4.0.1/code-server_4.0.1_amd64.deb
RUN dpkg -i code-server_4.0.1_amd64.deb
CMD [ "code-server", "--bind-addr", "0.0.0.0:10000" ]
