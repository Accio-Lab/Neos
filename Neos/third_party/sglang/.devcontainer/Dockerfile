FROM lmsysorg/sglang:dev

                                                 
                                                                                                                                                         
ARG HOST_UID=1003
ARG HOST_GID=1003
RUN groupadd -g $HOST_GID devuser && \
    useradd -m -u $HOST_UID -g $HOST_GID -s /bin/zsh devuser

                          
RUN apt-get update && apt-get install -y sudo && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devuser && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

                              
RUN cp -r /root/.oh-my-zsh /home/devuser/.oh-my-zsh && \
    cp /root/.zshrc /home/devuser/.zshrc && \
    cp /root/.vimrc /home/devuser/.vimrc && \
    cp /root/.tmux.conf /home/devuser/.tmux.conf && \
    sed -i 's|/root/.oh-my-zsh|/home/devuser/.oh-my-zsh|g' /home/devuser/.zshrc && \
    chown -R devuser:devuser /home/devuser/

                                       
WORKDIR /sgl-workspace/sglang
RUN chown -R devuser:devuser /sgl-workspace

                   
USER devuser

            
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

              
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
