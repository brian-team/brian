FROM debian/eol:stretch-slim
LABEL description="Legacy container for running Brian 1.4.3 with Python 2.7"
RUN apt-get update && \
    apt-get install -y -q --no-install-recommends \
        python-brian python-sympy python-nose ipython && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV MPLBACKEND=Agg
WORKDIR /workdir
