# Seminar-Poesie
Stripped to the Essence – The Disintegration of Poems through AI

This repository contains code and materials for a seminar project exploring
the differences between human-generated and AI-generated poetry, with a focus
on context, act, and artefact.
The project emphasizes reproducibility and fast deployment.

------------------------------------------------------------

REQUIREMENTS

General:
- Git

Local setup:
- Python >= 3.10
- pip

Docker setup (recommended):
- Docker
- Docker Compose

------------------------------------------------------------

PROJECT STRUCTURE

.
├── data/                   poems, datasets, generated artefacts
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   ├── analyzer.py         analysis / comparison logic
│   └── generator.py        poem generation logic
├── .gitignore
├── README.md
└── requirements.txt

------------------------------------------------------------

QUICK START (DOCKER – RECOMMENDED)

This is the fastest and most reproducible way to run the project.

git clone <REPO_URL>
cd poesie
docker compose -f docker/docker-compose.yml up --build

If the setup is correct, the container will start and execute the configured
entrypoint or demo logic.

To stop the container:

docker compose -f docker/docker-compose.yml down

------------------------------------------------------------

LOCAL SETUP (WITHOUT DOCKER)

1. Clone the repository

git clone <REPO_URL>
cd poesie

2. Create and activate a virtual environment (recommended)

python -m venv .venv
source .venv/bin/activate        (Linux / macOS)
.venv\Scripts\activate           (Windows)

3. Install dependencies

pip install -r requirements.txt

------------------------------------------------------------

START AI-POEM-GENERATION




------------------------------------------------------------

NOTES

- Docker is the preferred deployment strategy for evaluation.
- All experimental code lives in src/.
- data/ contains input poems, generated texts, and intermediate artefacts.
- The repository is intentionally kept minimal and transparent.

------------------------------------------------------------

LICENSE

Academic / educational use only.
