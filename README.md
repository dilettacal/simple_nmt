# Vanilla LSTM Neural Machine Translator with PyTorch

Seminarprojekt "Spezielle Anwendungen der Informatik - K.I. in der Robotik"
Bachelor of Science - Angewwandte Informatik HTW Berlin

Die vollständige Projektdokumentation befindet sich im Verzeichnis `documentation`.

Kurze Projektbeschreibung:

## 1. Projektstruktur
``` bash
.
├── data
│   ├── deu.txt             # Here deu.txt file should be placed
│   ├── prepro              # Stores all preprocessed pkl files
|── documentation           # Project documentation (Seminararbeit)
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

Um ein Experiment auszuführen, das Skript `run_experiment.py` vom Terminal starten.
Das Programm erlaubt folgende Einstellungen über die Konsole vorzunehmen:
1. Limit-Angabe `-l 50000`: Diese Angabe reduziert den Datensatz auf `limit` Exemplare. Standardmäßig wird der Datensatz auf 50000 Exemplare reduziert, da diese Einstellung die beste Ergebnisse geliefert hat.
2. Modell-Konfiguration:
2.1 embedding size `-e 256` 
2.2 hidden size `-h 256`
2.3 batch size `-b 64`
2.4 Anzahl der iterationen `-n 10000`

Jedes ausgeführte Experiment wird in der Datei `log_history.txt` geloggt. Das letzte Experiment wird in der Datei`last_experiment.txt` zusätzlich hinzugefügt.
Diese letzte Datei *muss nicht gelöscht* werden, da der Übersetzer auf die darin enthaltenen Informationen zugreifen muss, um ausgeführt zu werden!

Eine manuelle Änderung des letzten Experiments ist leider notwendig, falls ein anderes Experiment gestartet werden muss.


### 2.3 Übersetzer benutzen

Während der Experimentausführung werden Checkpoints in `experiment/checkpoints/<model_name>/<file_name>/<model_config>/*tar` gespeichert.
Am Ende des Experiments werden die wesentlichen Merkmale in der Datei `last_experiment.txt` gespeichert.

Um den Übersetzer aus der Konsole zu verwenden, soll folgender Befehl in der Konsole eingegeben werden:
`python translate.py`

Das Skript sucht nach dem letzten Experiment und prompt den Benutzer zum Eintippen.

Um den Übersetzer zu verlassen, `q` eingeben.

## 3. Implementierung

Anpassung der Implementierungen vom PyTorch Chatbot Tutorial und der Keras Implementierung aus Machine Learning Mastery (s. Quellen).

Übernahme in dieses Projekt:

Erweiterungen:

Verbesserungsvorschläge:

Was noch abgedeckt werden könnte:

## 4. Exemplarische Ergebnisse

## 5. Quellen

Papers/Bücher/Online-Beiträge:
- ...
- ...

Meist verwendete Code-Quellen sind:

- PyTorch Chatbot Tutorial, Chatbot Tutorial: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html 
- Machine Learning Mastery, How to Develop a Neural Machine Translation System from Scratch: https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/

Weitere Angaben befinden sich im Code.
