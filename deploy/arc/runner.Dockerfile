# Custom runner image: ARC default runner + CUDA toolkit + uv + python3.12.
# Build & push:
#   docker build -f deploy/arc/runner.Dockerfile \
#     -t ghcr.io/chakpongchung/mini-sglang-runner:latest .
#   docker push ghcr.io/chakpongchung/mini-sglang-runner:latest
#
# Image tag is referenced from deploy/arc/values.yaml.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG RUNNER_VERSION=2.321.0
ARG TARGETARCH=x64

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        jq \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        sudo \
        tar \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install uv (matches the install path used by the project's Dockerfile).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Set up the unprivileged runner account.
RUN useradd --create-home --shell /bin/bash --uid 1001 runner \
    && echo "runner ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/runner

WORKDIR /home/runner

# Install the GitHub Actions runner agent.
RUN curl -fL -o runner.tar.gz \
        https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-${TARGETARCH}-${RUNNER_VERSION}.tar.gz \
    && tar xzf runner.tar.gz \
    && rm runner.tar.gz \
    && ./bin/installdependencies.sh \
    && chown -R runner:runner /home/runner

USER runner

# ARC's RunnerScaleSet listener exec's run.sh — no ENTRYPOINT/CMD needed
# (values.yaml sets `command: ["/home/runner/run.sh"]`).
