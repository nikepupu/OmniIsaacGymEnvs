ARG ISAACSIM_VERSION=2023.1.1

FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION} as isaac-sim

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y neofetch unrar net-tools iputils-ping software-properties-common curl zip python3 python3-pip tmux git htop build-essential aria2 openssh-server xrdp xfce4* sudo vim libtf-dev && apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
    libatomic1 \
    libegl1 \
    libglu1-mesa \
    libgomp1 \
    libsm6 \
    libxi6 \
    libxrandr2 \
    libxt6 \
    wget \
&& apt-get -y autoremove \
&& apt-get clean autoclean \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN echo 'root:root' | chpasswd
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
RUN echo "xfce4-session" > /root/.xsession
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN cd /root && aria2c -q "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz" && tar xvf ngrok-v3-stable-linux-amd64.tgz && rm ngrok-v3-stable-linux-amd64.tgz
USER root
RUN cd /root && aria2c -q "https://go.microsoft.com/fwlink/?LinkID=760868" && DEBIAN_FRONTEND="noninteractive" apt install -y ./*.deb && rm *.deb && apt-get clean
RUN curl https://get.docker.com/ | bash
RUN usermod -aG docker root
RUN sed -ri 's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config
EXPOSE 1-65535

# Setup the required capabilities for the container runtime    
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# Open ports for live streaming
EXPOSE 47995-48012/udp \
       47995-48012/tcp \
       49000-49007/udp \
       49000-49007/tcp \
       49100/tcp \
       8011/tcp \
       8012/tcp \
       8211/tcp \
       8899/tcp \
       8891/tcp


# move the file here
RUN cd /root/ && ls && git clone https://github.com/nikepupu/OmniIsaacGymEnvs.git
RUN cd /root/ && mkdir .code

RUN cd /root/OmniIsaacGymEnvs/ && /isaac-sim/python.sh -m pip install -e .
