# TIDL 8.2 model compiler (matches BeagleBone AI-64 image: ti-tidl-8.2 / onnxruntime-tidl 1.7.0)
FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-setuptools wget ca-certificates libsm6 libxext6 libxrender1 libglib2.0-0 graphviz \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir -U "pip<22" && python3 -m pip install --no-cache-dir \
        "numpy==1.19.5" pyyaml "onnx==1.9.0" "protobuf<3.20" "pillow==8.2.0" "opencv-python-headless==4.2.0.34" \
        https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
WORKDIR /opt
RUN wget -q https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/tidl_tools.tar.gz \
    && tar -xzf tidl_tools.tar.gz && rm tidl_tools.tar.gz
ENV TIDL_TOOLS_PATH=/opt/tidl_tools LD_LIBRARY_PATH=/opt/tidl_tools
WORKDIR /work
