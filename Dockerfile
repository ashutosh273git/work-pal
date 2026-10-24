FROM python:3.14-slim 

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# make a dir for a code
WORKDIR /app

# pull the req file
COPY requirements.txt .

# run the req file
RUN uv pip install --system --no-cache -r requirements.txt

# copy my code
COPY . .

EXPOSE 8501

# how to run my app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]