#!/bin/bash
set -e

CUDA_VERSIONS="${1:-12-8,12-9}"

echo "==================================="
echo "Installing Docker..."
echo "==================================="

                                
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

                                    
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

                                  
sudo usermod -aG docker $USER

echo "Docker installed successfully!"
echo "Note: You need to log out and log back in for docker group membership to take effect"
echo ""

                                                
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
    BUILDER_NAME="pytorch/manylinux2_28-builder"
elif [ "$ARCH" = "aarch64" ]; then
    BUILDER_NAME="pytorch/manylinuxaarch64-builder"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

                                                    
echo "==================================="
echo "Pulling Docker Images..."
echo "==================================="
echo "Architecture: ${ARCH}"
echo "Builder: ${BUILDER_NAME}"

                                                          
IFS=',' read -ra CUDA_VERSION_ARRAY <<< "$CUDA_VERSIONS"

                                                                    
for CUDA_VERSION in "${CUDA_VERSION_ARRAY[@]}"; do
                     
    CUDA_VERSION=$(echo "$CUDA_VERSION" | xargs)

                                  
    CUDA_VERSION_DOTTED=$(echo "$CUDA_VERSION" | tr '-' '.')

    DOCKER_IMAGE="${BUILDER_NAME}:cuda${CUDA_VERSION_DOTTED}"

    echo ""
    echo "Pulling ${DOCKER_IMAGE}..."

                                                                                     
    if sg docker -c "docker pull ${DOCKER_IMAGE}"; then
        echo "✓ Successfully pulled ${DOCKER_IMAGE}"
    else
        echo "✗ Failed to pull ${DOCKER_IMAGE}"
        echo "  You may need to log out and log back in for docker group to take effect"
    fi
done

echo ""
echo "Docker images pulled successfully!"
echo ""

                            
if command -v lsb_release &> /dev/null; then
    UBUNTU_VERSION=$(lsb_release -rs | tr -d '.')
else
    UBUNTU_VERSION=$(. /etc/os-release && echo $VERSION_ID | tr -d '.')
fi

                                                                       
if [ "$ARCH" = "x86_64" ]; then
    CUDA_ARCH="x86_64"
elif [ "$ARCH" = "aarch64" ]; then
    CUDA_ARCH="sbsa"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

echo "==================================="
echo "System Information:"
echo "==================================="
echo "Ubuntu Version: ${UBUNTU_VERSION}"
echo "Architecture: ${ARCH}"
echo "CUDA Architecture: ${CUDA_ARCH}"
echo ""

                                                  
echo "==================================="
echo "Installing CUDA keyring..."
echo "==================================="
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/${CUDA_ARCH}/cuda-keyring_1.1-1_all.deb"
wget -q $KEYRING_URL -O cuda-keyring.deb
sudo dpkg -i cuda-keyring.deb
sudo apt-get update
rm cuda-keyring.deb
echo "CUDA keyring installed successfully!"
echo ""

                                          
IFS=',' read -ra CUDA_VERSION_ARRAY <<< "$CUDA_VERSIONS"

echo "==================================="
echo "Installing CUDA Toolkits..."
echo "==================================="
echo "Versions to install: ${CUDA_VERSIONS}"
echo ""

for CUDA_VERSION in "${CUDA_VERSION_ARRAY[@]}"; do
                     
    CUDA_VERSION=$(echo "$CUDA_VERSION" | xargs)

    echo "-----------------------------------"
    echo "Installing CUDA Toolkit ${CUDA_VERSION}..."
    echo "-----------------------------------"

    if sudo apt-get install -y cuda-toolkit-${CUDA_VERSION}; then
        echo "✓ CUDA Toolkit ${CUDA_VERSION} installed successfully!"
    else
        echo "✗ Failed to install CUDA Toolkit ${CUDA_VERSION}"
        echo "  This might be due to an invalid version or repository issue"
    fi
    echo ""
done

echo "==================================="
echo "Installation Summary"
echo "==================================="
echo "Installed CUDA versions:"
ls -d /usr/local/cuda-* 2>/dev/null || echo "No CUDA installations found in /usr/local/"
echo ""
echo "Setup complete!"
