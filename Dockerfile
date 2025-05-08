FROM ubuntu:24.10

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  build-essential clang cmake unzip wget \
  openjdk-23-jdk-headless \
  dotnet-sdk-9.0 \
  python3 python3-pip python3-setuptools \
  linux-tools-common linux-tools-generic \
  pkg-config \
  libc6-dev

RUN rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://sh.rustup.rs -O /tmp/rustup.sh \
  && sh /tmp/rustup.sh -y --no-modify-path \
  && rm /tmp/rustup.sh
ENV PATH="/root/.cargo/bin:${PATH}"

RUN cargo install --locked hyperfine

WORKDIR /app

CMD ["bash"]
