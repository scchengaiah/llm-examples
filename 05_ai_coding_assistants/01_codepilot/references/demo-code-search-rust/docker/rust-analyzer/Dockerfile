# Use an official Rust image as the base
FROM rust:latest
# Install rust-analyzer
RUN apt-get update && apt-get install -y curl \
    &&  curl -L https://github.com/rust-lang/rust-analyzer/releases/download/2024-09-02/rust-analyzer-x86_64-unknown-linux-gnu.gz \
    -o /tmp/rust-analyzer.gz \
    && ls -lh /tmp \
    && gunzip /tmp/rust-analyzer.gz \
    && ls -lh /tmp \
    && mv /tmp/rust-analyzer /usr/local/bin/rust-analyzer \
    && chmod +x /usr/local/bin/rust-analyzer

# Install LSIF generation tools
#RUN rustup toolchain install stable \
#    && rustup component add rust-analyzer

# Set environment variables
ENV QDRANT_PATH=/workspace/qdrant
ENV ROOT_PATH=/workspace

# Create necessary directories
RUN mkdir -p $QDRANT_PATH $ROOT_PATH/data

# Copy your code into the container (you can use a bind mount in docker-compose instead)
COPY . $QDRANT_PATH

# Set the working directory
WORKDIR $QDRANT_PATH

# Command to generate LSIF index
CMD ["sh", "-c", "/usr/local/bin/rust-analyzer -v lsif $QDRANT_PATH > $ROOT_PATH/data/index.lsif"]
