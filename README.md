# Rekurrente neuronale Netze in Pytorch am Beispiel eines simplen LSTM Maschinenübersetzers

Seminarprojekt "Spezielle Anwendungen der Informatik - K.I. in der Robotik"
Bachelor of Science - Angewwandte Informatik HTW Berlin


Kurze Projektbeschreibung:

## 1. Projekt
### 1.1 Projektziele

- Rekurrente neuronale Netze 
- Anwendung des Frameworks PyTorch in Zusammenhang mit RNNs
- Praxis: Implementierung eines simplen neuronalen Maschinenübersetzers


### 1.2 Projektstruktur

``` bash
.
├── data
│   ├── deu.txt             # Here deu.txt file should be placed
│   ├── prepro              # Stores all preprocessed pkl files
|── documentation           # Project documentation (Seminararbeit)
├── download.sh             # download dataset
├── experiment              # Stores experiment files (checkpoints, history, plots) and train_eval.py
│   ├── checkpoints         # Stores checkpoints and plots for every experiment
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
├── dry_run.py              # First experiments with standard settings
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
Dieser Datensatz kann aus der Webseite heruntergeladen werden. Dafür gibt es im Root-Verzeichnis das Skript `download.sh`. Das Skript lädt den zip-Ordner aus https://www.manythings.org/anki/ und kopiert die extrahierte Datei `deu.txt` in den `data`-Ordner. 

lternativ kann die zip-Datei aus dem Link manuell heruntergeladen werden. Die txt-Datei soll manuell in den Ordner `data` geschoben werden. Sollte die Dateiname nicht "deu.txt" heißen, so muss sie entsprechend umbenannt werden.

### 2.2 Packages
In der Datei `requirements.txt` sind die notwendigen Packages aufgelistet. Diese können in einem virtuellen Environment auch installiert werden.

### 2.3 Experiment ausführen

#### Ausführung mit `run_experiment.py`

Das ist die Variante mit Training und Validierungsphasen.
Um Experimente auszuführen soll das Skript `run_experiment.py` ausgeführt werden. Das Programm ist von der Konsole bedienbar. 

### Ausführung mit `dry_run.py`

Das ist die schnelle Variante mit Training und Valuierung über die Konsole (`translate.py`).

Folgende Argumente können verwendet werden:
1. `--limit`, z.B. --limit 50000: Limitiert die Exemplare auf 50000.
2. `--emb`, standardmäßig: 256: Die Anzahl der Features im Embedding-Layer
3. `--hid`, standardmäßig: 256: Die Anzahl der Hidden-Neurone im LSTM-Layer
4. `--batch_size`, standardmäßig 64
5. `--lr`,Learning-Rate, standardmäßig: 0.003
6. `--iterations`, Anzahl der Iterationen, standardmäßig 10000
7. `--teacher`, Teacher-Forcing-Ration, standardmäßig 1.0
8. `--nlayers`, Anzahl der gestackten LSTM-Einheiten, standardmäßig 1
9. `--max_len`, Sequenzlänge, standardmäßig 10

Weitere Argumente können über: `python run_experiment.py --help` angesehen werden.

Jedes ausgeführte Experiment wird in der Datei `log_history.txt` geloggt. Das letzte Experiment wird in der Datei`last_experiment.txt` zusätzlich hinzugefügt.
Diese letzte Datei *muss nicht gelöscht* werden, da der Übersetzer auf die darin enthaltenen Informationen zugreifen muss, um ausgeführt zu werden.

**CUDA-Hinweis**: 
Die Verwendung der GPU wird global im System verwaltet (`global_settings.py`). Zu Experimentenbeginn wird geprüft, ob CUDA verfügbar ist. Wenn das der Fall ist, dann wird der Device auf **"cuda"** automatisch gesetzt. 


### 2.4 Übersetzer benutzen (`translate.py`)

Während der Experimentausführung werden Checkpoints in `experiment/checkpoints/<model_name>/<file_name>/<model_config>/checkpoint.tar` gespeichert.
Um den Übersetzer zu starten, soll das Skript `translate.py` ausgeführt werden.

Beispielanwendung:
1. `python translate.py`: Greift auf das letzte Experiment, das in der Datei `last_experiment.txt` gespeichert wird. Der Übersetzer wird im Konsole-Modus gestartet.
2. `python translate.py --file 'True'`: Greift auf das letzte Experiment, das in der Datei `last_experiment.txt` gespeichert wird. Es werden Beispielübersetzungen in einer Datei im Experimentenordner gespeichert.
3. `python translate.py --file 'True' --path path_to_experiment`: Greift auf das Experiment, das mit Path übergeben wird und schreibt Übersetzungen in die Datei.
4. `python translate.py  --path path_to_experiment`: Greift auf das Experiment, das mit Path übergeben wird und startet das Experiment in der Konsole

Um den Übersetzer zu verlassen, `q` eingeben.

Beispiel-Aufruf:
```bash
python translate.py --path experiment/checkpoints/dry_run_simple_nmt_model_full_158544_teacher_1.0_train_voc_adam_lr-0.001-1/deu.txt/2-2_512-512_100

```

## 3. Exemplarische Ergebnisse

* Bestes Ergebnis erzielt mit `dry_run.py`*:

```bash
python dry_run.py --tbptt "False" --max_len 10 --emb 512 --hid 512 --teacher 1.0 --iterations 15000 --batch_size 100 --lr 0.001 --dec_lr 1 --nlayers 2 
```

* Gutes Ergebnis erzielt mit `run_experiment.py`* :
```bash
python run_experiment.py --tbptt "False" --max_len 10 --emb 512 --hid 512 --teacher 1.0 --iterations 30000 --batch_size 100 --lr 0.003 --dec_lr 1 --nlayers 2
```

Beispielübersetzungen :

| Source        | Target           
| ------------- |-------------
| the woman is reading     | die frau liest gerade 
| the man is cooking      | der mann kocht kochen      
| the story is too long | die geschichte ist zu lange
| the train has arrived | der zug ist da
| the train has left | der zug ist abgefahren
| the train has already left | der zug ist schon abgefahren
| their poems are good | ihre gedichte sind gut
| i think you should stop screaming | ich finde du sollten aufhoeren zu schreien
| I think you should stop writing your essay | ich denke du solltest deinen aufsatz zu stoeren


## 5. Quellen

Code-Quellen:

- PyTorch Chatbot Tutorial, Chatbot Tutorial: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html 
- Machine Learning Mastery, How to Develop a Neural Machine Translation System from Scratch: https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/

Weitere Angaben befinden sich im Code.
