
FROM python:3.7-slim

RUN apt-get update

#copy files
COPY . ./

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the streamlit on container startup
CMD [ "streamlit", "run","app.py" ]