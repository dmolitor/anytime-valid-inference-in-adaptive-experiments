FROM rocker/binder:4
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --chown=${NB_USER} . /home/rstudio
WORKDIR /home/rstudio
USER root
RUN printf "RETICULATE_PYTHON_ENV=/home/rstudio/.venv\nVIRTUAL_ENV=/home/rstudio/.venv\n" >> /usr/local/lib/R/etc/Renviron.site

USER ${NB_USER}
RUN test -f pyproject.toml || uv init --app . || true
RUN uv sync --all-groups --all-extras
RUN uv add -r requirements-tugboat.txt