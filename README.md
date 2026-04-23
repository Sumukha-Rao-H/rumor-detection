# Stock Market Rumor Detection System

An RL-based system that scrapes financial subreddits, correlates posts with real stock price movements, and classifies them as rumors, leaks, or confirmed news.

## Project Structure

```
.
├── reddit_data_fetcher.py   # Reddit + yfinance scraper (psycopg2 bulk inserts)
├── models.py                # SQLAlchemy ORM models for all 4 tables
├── database.py              # SQLAlchemy engine + session setup
├── features.py              # Sample feature engineering query for RL training
├── rl_model.py              # Reinforcement Learning model architecture
├── reward.py                # Reward function(s) for the RL agent
├── train.py                 # Training loop
├── Schema.sql               # Raw SQL schema (used by Docker for auto-init)
├── alembic.ini              # Alembic config
├── alembic/
│   ├── env.py               # Reads DB URL from .env
│   ├── script.py.mako       # Migration template
│   └── versions/            # Migration files
├── docker-compose.yml       # Postgres 16 + pgAdmin 4
├── start-db.sh              # Single-container startup script
├── .env.example             # Template for environment variables
└── requirements.txt         # Python dependencies
```

## Prerequisites

- Python 3.10+
- Docker (and Docker Compose)

## Setup

### 1. Clone and configure environment

```bash
git clone <repo-url> && cd <repo-dir>

# Copy the example env and fill in your values
cp .env.example .env
```

Edit `.env` with your preferred credentials:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_rumors
DB_USER=postgres
DB_PASSWORD=<your-secure-password>
PGADMIN_EMAIL=admin@admin.com
PGADMIN_PASSWORD=admin
```

### 2. Create a Python virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
# venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the database

**Option A -- Docker Compose** (Postgres + pgAdmin):

```bash
docker compose up -d
```

pgAdmin will be available at [http://localhost:5050](http://localhost:5050).

**Option B -- Single container** (Postgres only):

```bash
chmod +x start-db.sh
./start-db.sh
```

### 5. Run database migrations

```bash
alembic upgrade head
```

This creates all four tables (`reddit_posts`, `stock_prices`, `post_ticker_links`, `post_labels`) along with indexes and constraints.

## Usage

### Collect data

Scrape posts from r/wallstreetbets, r/stocks, r/investing, and r/StockMarket, then fetch 7 days of OHLCV price data for every ticker found:

```bash
python reddit_data_fetcher.py
```

### Preview training features

Join posts with stock prices and labels to generate training-ready rows for the RL model:

```bash
python features.py
```

### Train the RL model

```bash
python train.py
```

## Working with Migrations

After changing models in `models.py`, generate a new migration:

```bash
# Requires a running database
alembic revision --autogenerate -m "describe your change"

# Apply it
alembic upgrade head

# Roll back one step
alembic downgrade -1
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Subreddits | r/wallstreetbets, r/stocks, r/investing, r/StockMarket |
| Stock data | yfinance |
| Database | PostgreSQL 16 (Docker) |
| ORM | SQLAlchemy 2.0 |
| Migrations | Alembic |
| Bulk inserts | psycopg2 + execute_values |
| Admin UI | pgAdmin 4 |
