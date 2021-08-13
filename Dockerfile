FROM python:3.6.1-alpine 

WORKDIR /Speech-Recognition 

ADD . /Speech-Recognition 

RUN pip install -r requirements.txt 

EXPOSE 5000

CMD ["python","app.py"]  
