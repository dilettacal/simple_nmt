# Vanilla LSTM Neural Machine Translator with PyTorch

Seminarprojekt "Spezielle Anwendungen der Informatik - K.I. in der Robotik"
Bachelor of Science - Angewwandte Informatik HTW Berlin

Kurze Projektbeschreibung:

## 1. Projektstruktur
``` bash
.
├── data
│   ├── deu.txt             # Here deu.txt file should be placed
│   ├── prepro              # Stores all preprocessed pkl files
├── download.sh             # download script
├── experiment              # Stores experiment files (checkpoints, history, plots) and train_eval.py
│   ├── checkpoints
│   │   ├── plots
│   ├── log_history.txt
│   └── train_eval.py
├── global_settings.py      # defines global settings
├── model                   # Model components
│   ├── model.py            
├── notebooks               # Jupyter notebooks
│   └── Presentation.ipynb  
├── Pipfile         
├── Pipfile.lock
├── README.md
├── run_experiment.py       # main execution file
├── translate.py            # translate.py
├── tutorial                # tutorial
└── utils                   # utilities, e.g. mappings, preprocessing, tokenization, general utils
    ├── mappings.py
    ├── prepro.py
    ├── tokenize.py
    └── utils.py
```

## 2. Programm verwenden

### 2.1 Datensatz herunterladen
Im Projekt wird der Tatoeba-Datensatz für Deutsch-Englisch verwendet.
Dieser Datensatz kann aus der Webseite heruntergeladen werden.
Dafür gibt es im Root-Verzeichnis das Skript `download.sh`. Das Skript lädt den zip-Ordner aus https://www.manythings.org/anki/ und kopiert die extrahierte Datei `deu.txt` in den `data`-Ordner. 
Alternativ kann die zip-Datei aus dem Link manuell heruntergeladen werden. Die txt-Datei soll manuell in den Ordner `data` geschoben werden.

### 2.2 Experiment ausführen
### 2.3 Übersetzer benutzen

## 3. Hinweis zur Implementierung
## 4. Quellen
