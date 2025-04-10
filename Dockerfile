# Use an official Python image with OpenJDK
FROM python:3.11-slim

# Install Java (for H2O)
RUN apt-get update && apt-get install -y openjdk-11-jdk curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME so H2O works
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run Streamlit app
CMD ["streamlit", "run", "foster-care-displacement-pred.py", "--server.port=8000", "--server.address=0.0.0.0"]
