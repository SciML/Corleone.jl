# Use Julia 1.10 as base image
FROM julia:1.10
# Copy project files into the container
COPY . /lib
# Set working directory
WORKDIR /lib
