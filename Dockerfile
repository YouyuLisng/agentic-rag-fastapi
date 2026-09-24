# ---- Stage 1: build the virtualenv ----
# `ghcr.io/astral-sh/uv` ships just the uv binary -- copying it in
# avoids needing pip to install uv first.
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only the dependency manifests first. Docker caches each layer by
# its inputs -- as long as pyproject.toml/uv.lock don't change, this
# `uv sync` layer is reused on every rebuild, even after editing code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Now copy the rest of the app. Changing app code only invalidates this
# layer onward, not the (usually much slower) dependency install above.
COPY . .

# ---- Stage 2: runtime ----
# A fresh base image, not the builder -- the final image never sees uv
# itself, pip's cache, or anything else only needed to build the venv.
FROM python:3.12-slim AS runtime
WORKDIR /app

COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

# Don't run as root inside the container -- if the app is ever
# compromised, a non-root process can't touch anything outside what it
# was explicitly given.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Shell form (not exec form) so $PORT actually expands -- platforms
# like Render assign a dynamic port via this env var and route traffic
# to whatever the service actually listens on, ignoring EXPOSE/any
# hardcoded port. Defaults to 8000 for docker-compose/local runs,
# which don't set PORT at all. Calls uvicorn directly (not `uv run
# uvicorn ...`) -- the venv is already fully built and on PATH, so
# there's nothing left for uv to resolve or sync at container start.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
