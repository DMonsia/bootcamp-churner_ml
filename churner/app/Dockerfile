FROM --platform=linux/amd64 python:3.8-slim
WORKDIR /usr/src/app
COPY ./requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt &&\
    rm -r /tmp/requirements.txt
RUN groupadd -g 1000 mlgroup && useradd -u 1000 -g mlgroup mluser
RUN chown mluser:mlgroup /usr/src/app
USER mluser:mlgroup
COPY --chown=mluser:mlgroup . .
EXPOSE 5555
ENTRYPOINT [ "python", "api.py" ]