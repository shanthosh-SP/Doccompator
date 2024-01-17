FROM centos:8.3.2011

# Fixed Failed to download metadata for repo 'appstream': Cannot prepare internal mirrorlist: No URLs in mirrorlist
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-Linux-* &&\
    sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-*

RUN yum upgrade -y

## Install NVM and use Node version specified in .npmrc
RUN	curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash

COPY .nvmrc .

RUN source ~/.bashrc && nvm install --lts

# Install Angular CLI - Do not send anonymous stats to google
RUN source ~/.bashrc && echo 'N' | npm install -g @angular/cli

WORKDIR /usr/src

COPY package.json .

# answer 'yes' to confirm installation
RUN source ~/.bashrc && echo 'y' | npm install --force

WORKDIR '/frontend'

RUN ln -s /usr/src/node_modules node_modules

ENTRYPOINT [""]
