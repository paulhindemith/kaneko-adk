FROM python:3.12.11-slim-bullseye

WORKDIR /app

RUN pip install pipx && python -m pipx ensurepath

RUN pipx install poetry

ENV PATH="/root/.local/bin:${PATH}"

COPY poetry.lock pyproject.toml README.md ./

RUN poetry install --no-root --without dev

COPY --from=kaneko-adk kaneko_adk kaneko_adk

COPY tables tables

COPY main.py .

EXPOSE 8080

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]