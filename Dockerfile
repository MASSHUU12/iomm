FROM fedora:42

RUN dnf install -y \
  perf \
  clang cmake unzip wget \
  java-21-openjdk-portable \
  dotnet-sdk-9.0 \
  python3 python3-pip python3-setuptools \
  pkg-config \
  glibc-devel \
  time \
  gnuplot \
  && dnf clean all

RUN rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://sh.rustup.rs -O /tmp/rustup.sh \
  && sh /tmp/rustup.sh -y --no-modify-path \
  && rm /tmp/rustup.sh
ENV PATH="/root/.cargo/bin:${PATH}"

RUN cargo install --locked hyperfine

WORKDIR /app

CMD ["bash"]
