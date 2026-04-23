#!/usr/bin/env bash
# Use this script to start a docker container for the local development database
#
# TO RUN ON WINDOWS:
# 1. Install WSL - https://learn.microsoft.com/en-us/windows/wsl/install
# 2. Install Docker Desktop - https://docs.docker.com/docker-for-windows/install/
# 3. Open WSL - `wsl`
# 4. Run this script - `./start-database.sh`
#
# On Linux and macOS: `./start-database.sh`

DB_NAME="stock_rumors"
DB_CONTAINER_NAME="${DB_NAME}-postgres"

# ── Check Docker ─────────────────────────────────────────────
if ! [ -x "$(command -v docker)" ]; then
  echo -e "Docker is not installed. Please install docker and try again.\nDocker install guide: https://docs.docker.com/engine/install/"
  exit 1
fi

# ── Already running? ─────────────────────────────────────────
if [ "$(docker ps -q -f name=$DB_CONTAINER_NAME)" ]; then
  echo "Database container '$DB_CONTAINER_NAME' already running"
  exit 0
fi

# ── Exists but stopped? ──────────────────────────────────────
if [ "$(docker ps -q -a -f name=$DB_CONTAINER_NAME)" ]; then
  docker start "$DB_CONTAINER_NAME"
  echo "Existing database container '$DB_CONTAINER_NAME' started"
  exit 0
fi

# ── Load .env ────────────────────────────────────────────────
if [ ! -f .env ]; then
  echo ".env file not found. Copying from .env.example..."
  cp .env.example .env
fi

set -a
source .env
set +a

# ── Read vars ────────────────────────────────────────────────
DB_PASSWORD="${DB_PASSWORD:-password}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-postgres}"

# ── Warn about default password ──────────────────────────────
if [ "$DB_PASSWORD" = "password" ]; then
  echo "You are using the default database password."
  read -p "Should we generate a random password for you? [y/N]: " -r REPLY
  if ! [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Please set DB_PASSWORD in your .env file and try again."
    exit 1
  fi
  DB_PASSWORD=$(openssl rand -base64 12 | tr '+/' '-_')
  # Update DB_PASSWORD= line in .env
  sed -i -e "s#^DB_PASSWORD=.*#DB_PASSWORD=${DB_PASSWORD}#" .env
  echo "Random password written to .env"
fi

# ── Start container ──────────────────────────────────────────
docker run -d \
  --name "$DB_CONTAINER_NAME" \
  -e POSTGRES_USER="$DB_USER" \
  -e POSTGRES_PASSWORD="$DB_PASSWORD" \
  -e POSTGRES_DB="$DB_NAME" \
  -p "$DB_PORT":5432 \
  -v "${DB_CONTAINER_NAME}_data":/var/lib/postgresql/data \
  docker.io/postgres:16-alpine \
  && echo "Database container '$DB_CONTAINER_NAME' was successfully created"