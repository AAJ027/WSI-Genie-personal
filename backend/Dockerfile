FROM continuumio/miniconda3
WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
RUN apt-get update && apt-get install -y fuse

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "genie", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
# RUN echo "Make sure flask is installed:"
# RUN python -c "import flask"

# Copy gunicorn config
COPY gunicorn.conf.py .
# Copy project code
# code must go into backend folder for file pathing reasons
COPY . /app/backend

EXPOSE 8000/tcp
# The code to run when container is started:
# Runs gunicorn server from parent of files, will automatically find gunicorn.conf.py
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "genie", "gunicorn", "-b", "0.0.0.0:8000", "backend.app:app"]