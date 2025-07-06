FROM fedora:42

RUN dnf install -y \
  perf \
  unzip wget \
  python3 python3-pip python3-setuptools \
  pkg-config \
  glibc-devel \
  time \
  gnuplot \
  golang \
  zig \
  && dnf clean all

ENV GOPATH=/root/go
ENV PATH="$GOPATH/bin:${PATH}"

RUN go install golang.org/x/perf/cmd/benchstat@latest

RUN rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://sh.rustup.rs -O /tmp/rustup.sh \
  && sh /tmp/rustup.sh -y --no-modify-path \
  && rm /tmp/rustup.sh
ENV PATH="/root/.cargo/bin:${PATH}"

RUN cargo install --locked hyperfine

WORKDIR /app

CMD ["bash"]
