FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3.8 \
    p7zip-full


#RUN apt-get install python3.8 -y
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Installing python dependencies
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

# Copy app files
RUN mkdir /app
COPY . /app
WORKDIR /app/
RUN gdown https://drive.google.com/uc?id=1UpPoMB2Ke91xX8oUrKicFlDp-7sZAkbo \
&& 7za x tasks.7z
#RUN 7za x tasks.zip
ENTRYPOINT ["python3", "script.py"]