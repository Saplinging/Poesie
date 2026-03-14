# Seminar-Poesie

Projekt fuer ein Seminar zur Gegenueberstellung von menschlicher und KI-generierter Lyrik.
Die Daten liegen als JSON-Dateien vor, die Generierung erfolgt ueber die OpenAI API,
und die Auswertung schreibt Metriken, Tabellen und Plots nach `results/`.

## Was dieses Repository tut

- `src/generator.py` liest Prompt-Dateien aus `data/prompt/` und erzeugt dazu KI-Gedichte in `data/ai/`.
- `src/analyzer.py` vergleicht `data/human/` und `data/ai/` und schreibt Ausgaben nach `results/`.
- Das Docker-Setup ist ein Dev-Container-Setup. Es startet nicht automatisch Generator oder Analyzer.

## Projektstruktur

```text
.
|- data/
|  |- ai/          erzeugte KI-Gedichte als JSON
|  |- human/       menschliche Referenzgedichte als JSON
|  `- prompt/      Prompt-Dateien fuer die Generierung
|- docker/
|  |- Dockerfile
|  `- docker-compose.yml
|- results/        Analyseausgaben, Tabellen, Plots
|- src/
|  |- analyzer.py
|  `- generator.py
|- README.md
`- requirements.txt
```

## Voraussetzungen

### Lokal

- Python 3.10 oder neuer
- Git
- Eine funktionierende `pip`-Installation
- Fuer `src/generator.py`: ein OpenAI API Key

### Docker

- Docker Desktop oder Docker Engine
- Docker Compose v2

## Lokales Setup

### 1. Repository klonen

```powershell
git clone <REPO_URL>
cd Poesie
```

### 2. Virtuelle Umgebung anlegen

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Abhaengigkeiten installieren

```powershell
pip install -r requirements.txt
```

Hinweis: Das Setup ist eher schwergewichtig. `torch`, `transformers`, `spacy`
und `perplexFern` ziehen viele Pakete nach.

### 4. OpenAI API Key setzen

Option A: `.env` im Projektwurzelverzeichnis anlegen:

```env
OPENAI_API_KEY=dein_key_hier
```

Option B: nur fuer die aktuelle Shell setzen.

PowerShell:

```powershell
$env:OPENAI_API_KEY="dein_key_hier"
```

macOS / Linux:

```bash
export OPENAI_API_KEY="dein_key_hier"
```

## Docker-Setup

Das Docker-Setup wurde als Dev-Container geprueft.
Folgendes funktioniert:

- `docker compose -f docker/docker-compose.yml config`
- `docker compose -f docker/docker-compose.yml build`
- `docker compose -f docker/docker-compose.yml run --rm poesie-dev python --version`
- `docker compose -f docker/docker-compose.yml up -d`
- `docker compose -f docker/docker-compose.yml down`

Wichtig:

- `docker compose up` startet nur den Dev-Container.
- Generator und Analyzer laufen nicht automatisch los.
- Der Build dauert spuerbar, weil unter anderem `torch` mit installiert wird.

### Docker-Container bauen

```powershell
docker compose -f docker/docker-compose.yml build
```

### Interaktive Shell im Container

```powershell
docker compose -f docker/docker-compose.yml run --rm poesie-dev bash
```

### Container im Hintergrund starten

```powershell
docker compose -f docker/docker-compose.yml up -d
docker exec -it poesie-dev bash
```

Danach wieder aufraeumen:

```powershell
docker compose -f docker/docker-compose.yml down
```

## Bedienung

### KI-Gedichte generieren

`src/generator.py` ist OpenAI-only. Es gibt keinen lokalen HuggingFace-Fallback mehr.

Standardaufruf lokal:

```powershell
python src/generator.py --prompt-dir data/prompt --ai-dir data/ai --model gpt-4o-mini --num-samples 10
```

Wichtige Punkte:

- Es wird pro Prompt eine `ai_<poem-id>.json` geschrieben.
- Es werden standardmaessig 10 unabhaengige Samples erzeugt.
- Bestehende Dateien in `data/ai/` werden nur mit `--overwrite` ersetzt.

Nutzliche Optionen:

- `--prompt-dir`: anderer Prompt-Ordner
- `--ai-dir`: anderes Zielverzeichnis
- `--model`: anderes OpenAI-Modell
- `--api-key`: API Key direkt uebergeben statt `.env`
- `--overwrite`: existierende `ai_*.json` ersetzen
- `--num-samples`: Anzahl der Samples pro Prompt

Docker-Variante:

```powershell
docker compose -f docker/docker-compose.yml run --rm poesie-dev python src/generator.py --prompt-dir data/prompt --ai-dir data/ai --model gpt-4o-mini --num-samples 10
```

### Analyse ausfuehren

Standardaufruf lokal:

```powershell
python src/analyzer.py --human-dir data/human --ai-dir data/ai --prompt-dir data/prompt --results-dir results
```

Was dabei entsteht:

- `results/metrics.csv`
- `results/metrics.json`
- `results/metrics_pairs.csv`
- `results/metrics_samples.csv`
- `results/plots/*`
- `results/perplexfern_outputs/*`

Nutzliche Optionen:

- `--results-dir`: anderes Zielverzeichnis fuer neue Ergebnisse
- `--no-ppl`: Perplexity-Berechnung abschalten
- `--no-perplexfern-images`: keine perplexFern-Bilder erzeugen
- `--perplexfern-metrics perplexity,entropy,ttr`: Metriken einschranken

Docker-Variante:

```powershell
docker compose -f docker/docker-compose.yml run --rm poesie-dev python src/analyzer.py --human-dir data/human --ai-dir data/ai --prompt-dir data/prompt --results-dir results
```

## Was Generator und Analyzer veraendern

- `src/generator.py` schreibt oder ueberschreibt Dateien in `data/ai/`.
- `src/analyzer.py` schreibt oder ueberschreibt Dateien in `results/`.
- Keines der Skripte startet implizit durch `docker compose build` oder `docker compose up`.

Wenn du bestehende Ergebnisse behalten willst:

- Generator ohne `--overwrite` laufen lassen
- Fuer neue Analyse-Laeufe ein anderes `--results-dir` waehlen

## Format der erzeugten AI-JSONs

Der Generator schreibt JSON-Dateien im Multi-Sample-Format:

- Top-Level-Felder: `ai-poem-id`, `human-poem-id`, `type`, `generator`, `samples`, `text`, `texts`
- `samples` enthaelt 10 Einzelobjekte mit `sample-id`, `index`, `generator`, `text`
- `text` ist das erste Sample
- `texts` enthaelt alle Sample-Texte in Reihenfolge

Die `generator`-Metadaten enthalten auf Top-Level und pro Sample:

- Zeitstempel
- Prompt-ID
- Backend
- Modell
- Temperatur
- `max_tokens`

## Troubleshooting

### `OPENAI_API_KEY not set`

Setze den Key in `.env` oder als Umgebungsvariable.

### Docker-Build dauert sehr lange

Das ist in diesem Projekt erwartbar. Die Analyse-Abhaengigkeiten ziehen grosse
Pakete nach, besonders `torch`.

### Bestehende AI-JSONs oder Ergebnisse sollen nicht veraendert werden

- Generator nicht mit `--overwrite` starten
- Analyzer auf ein neues Zielverzeichnis schreiben, zum Beispiel:

```powershell
python src/analyzer.py --human-dir data/human --ai-dir data/ai --prompt-dir data/prompt --results-dir results_run_02
```
