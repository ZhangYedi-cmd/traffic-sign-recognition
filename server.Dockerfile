FROM python:3.9

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

#WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 时区设置
RUN /bin/cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
&& echo 'Asia/Shanghai' >/etc/timezone \

EXPOSE 8080
COPY .  .
#
ENTRYPOINT ["gunicorn", "--config", "gunicorn.py", "server:app"]