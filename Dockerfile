FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY configs ./configs
COPY data/fixtures ./data/fixtures
COPY vendor/pyFDS-Evac ./vendor/pyFDS-Evac
COPY index.html app.js styles.css overrides.css ./
COPY station_viewer.js station_base.css ./
COPY private_inspector.js ./
COPY gaehwasan.html gaehwasan.js gaehwasan.css ./
COPY gaehwasan3d.js ./
COPY vendor/three ./vendor/three
EXPOSE 8000
CMD ["uvicorn", "persona_evac.visualization.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS test
RUN pip install --no-cache-dir '.[dev]'
COPY tests ./tests
COPY scripts ./scripts
COPY pages_replay.js ./
CMD ["python", "-m", "pytest"]

# The upstream simulator is kept in its own image because its scientific stack
# is substantially heavier than the API/UI runtime.
FROM base AS pyfds
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir ./vendor/pyFDS-Evac
ENTRYPOINT ["python", "vendor/pyFDS-Evac/run.py"]

# FDS 6.11.1 is intentionally isolated from the web image. Build this target
# only when a real FDS deck must be precomputed.
FROM ubuntu:24.04 AS fds
ARG FDS_VERSION=6.11.1
ARG SMV_VERSION=6.11.2
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/fds.sh "https://github.com/firemodels/fds/releases/download/FDS-${FDS_VERSION}/FDS-${FDS_VERSION}_SMV-${SMV_VERSION}_lnx.sh" \
    && cd /tmp \
    && printf '2\n' | bash /tmp/fds.sh \
    && mkdir -p /opt/fds \
    && tar -xzf "FDS-${FDS_VERSION}_SMV-${SMV_VERSION}_lnx.tar.gz" -C /opt/fds \
    && find /opt/fds -type f -name fds -exec ln -s {} /usr/local/bin/fds \; \
    && rm /tmp/fds.sh "/tmp/FDS-${FDS_VERSION}_SMV-${SMV_VERSION}_lnx.tar.gz"
ENV LD_LIBRARY_PATH=/opt/fds/FDS-6.11.1_SMV-6.11.2_lnx/bin/intelmpi/lib
ENV I_MPI_FABRICS=shm
COPY vendor/pyFDS-Evac/assets/t_junction/t_junction.fds /input/t_junction.fds
# FDS 6.11.1 rejects the upstream deck's rounded oxygen coefficient by 1e-4.
# Correct only the image copy; the pinned upstream submodule remains untouched.
RUN sed -i 's/-1.4377,/-1.43775,/' /input/t_junction.fds \
    && sed -i 's/T_END=300/T_END=60/' /input/t_junction.fds
WORKDIR /work
ENTRYPOINT ["/bin/bash", "-lc"]
