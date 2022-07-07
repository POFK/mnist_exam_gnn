FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
MAINTAINER TX Mao <mtianxiang@gmail.com>

ARG UNAME=mtx
ARG UID=1001
ARG GID=1001

RUN groupadd -g $GID -o $UNAME
RUN useradd -r -u $UID -g $GID -o -d /home/$UNAME -s /bin/bash -p '$1$TRYvpNbr$XfFU1QixEe4rup6g7izUU.' $UNAME && adduser $UNAME sudo

RUN apt update -y &&\
    apt install -y sudo vim proxychains

ADD ./ /home/$UNAME/code
RUN mkdir /home/$UNAME/data
RUN chown -R $UNAME /home/$UNAME && chgrp -R $UNAME /home/$UNAME

USER $UNAME

RUN pip install -r /home/$UNAME/code/requirements.txt

WORKDIR /home/$UNAME
