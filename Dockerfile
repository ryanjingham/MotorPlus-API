FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables
ENV FLASK_APP=API.py
ENV FLASK_ENV=development

# Expose port 5000 for Flask to listen on
EXPOSE 5000

# Define the command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]