FROM mcr.microsoft.com/azureml/curated/minimal-ubuntu20.04-py38-cuda11.6.2-gpu-inference:13

WORKDIR /ssast_ciab

ADD . /ssast_ciab

RUN pip install -r ssast_ciab/requirements.txt 

CMD [ "streamlit" , "run" , "/workspace/app.py", "--server.address=0.0.0.0" ]

RUN chown -R 42420:42420 /ssast_ciab

ENV HOME=/ssast_ciab