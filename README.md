# Vanilla LSTM Neural Machine Translator with PyTorch

Seminarprojekt "Spezielle Anwendungen der Informatik - K.I. in der Robotik"
Bachelor of Science - Angewwandte Informatik HTW Berlin

Die vollständige Projektdokumentation befindet sich im Verzeichnis `documentation`.

Kurze Projektbeschreibung:

## 1. Projekt
### 1.1 Projektziele

- Rekurrente neuronale Netze kennenlernen
- Implementierung eines Vanilla LSTM Neuronalen Maschinenübersetzers
- Anwendung des Frameworks PyTorch in Zusammenhang mit RNNs

Erwartete Ergebnisse:

Das Modell ist in der Lage, kurze Sätze zu übersetzen, wie z.B. "I am an engineer", "I live near you", "I love you", "how are you?", "what do you do?"


### 1.2 Projektstruktur

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

Phasen:

1. Preprocessing: Hier wird der Datensatz so bereinigt, dass bei der Verwendung nur eine Tokenisierung mit der nativen python Funktion `split(" )` ausreicht.
2. Dictionary: Für jede Sprache wird ein Objekt `Voc` erstellt. Es enthält ein Mapping zwischen den Wörtern im Datensatz und ihrer Indexposition. Das erlaubt die Umwandlung zwischen Strings und numerische Darstellung. Das Dictionary wird auf dem Trainingdatensatz gebildet.
3. Datensatz gesplittet auf drei Datensätze: 70% Train, 20% Validation, 10% Test
4. Modellkomponente:
    - Hyperparameter: 
        - hidden size (Anzahl der Neuronen in einer LSTM-Einheit), 
        - embedding size (Dimensionalität der Wortdarstellung im Vektorraum), 
        - n_layers (Anzahl der Layers im Modell)
    - Feste Parameter: Input und output size. Diese beziehen sich jeweils auf die Vocabulary-Größe für die Input- und die Targetsprachen
    - Encoder: input size, hidden size, embedding size 
        - Embedding Layer(input_size, embedding_size)
        - LSTM-Einheit (embedding_size, hidden_size)
    - Decoder: output size, hidden size, embedding size
        - Embedding (output_size, embedding_size)
        - LSTM-Einheit (embedding_size, hidden_size)
        - Linear Layer (hidden_size, output_size)
        - Log-Softmax auf Ergebnis des Linear Layers
5. Training: Batch-Training auf Training- und Validation-Dataset
    - Hyperparameter:
        - Learning rate: Default 0,0001
        - Gradient clipping
        - Batch size: Default 64
        - Iterationen: 10000 (default)
6. Evaluation: Auf Testdatensatz
7. Logging: 
    - Jedes Experiment geloggt und geplottet 

Übernahme in dieses Projekt:

- Trainingsverfahren
- Vocabulary-Klasse
- Vektorisierungsverfahren (Umwandlung Token > Indizes und umgekehrt)
- Behandlung der Batches und des Paddings (Maskfunktion für die Loss-Berechnung)
- Struktur des Encoders und Decoders (Der Decoder wurde downgraded zu einem Vanilla-Decoder)
- Funktionen zum Speichern/Laden eines Modells
- Evaluation aus Tastatur

Erweiterungen:

- Preprocessing verbessert
- Statt nur eine Sprache (ausreichend für einen Chatbot), zwei Sprachen berücksichtigt
- Keine besondere Filter auf den Sätzen
- Batch-Training erweitert auf Training, Validation und Test Dataset
- Allgemeine Programmstruktur deutlicher gemacht
- Plotting

Verbesserungsvorschläge:

- Batch training sollte verbessert werden, etwa durch die Verwendung der PyTorch-Datenstrukturen: `Dataset` und `DataLoader`.
    - Dabei muss aber die unterschiedliche Sequenzlänge berücksichtigt und für den `DataLoader` durch Überschreiben der `collate_fn` behandelt werden. Ansosnten verkürzt der DataLoader automatisch alle Daten im Batch auf die kürzeste Inputsequenz.
- Padding und Masking der Loss-Funktion:
    - PyTorch bietet zwei Packages für Loss-Funktionen:
        - `nn.functional`: Enthält Funktionen, wie z.B. `log_softmax` in Kombination mit `NLLLoss`
        - `nn`: Enthält Objekte, bzw. Wrapper um Loss-Funktionen, wie z.B. `CrossEntropyLoss`. Ab der neusten Version berechnet das `CrossEntropyLoss`intern sowohl `log_softmax`, als auch `NLLLoss`. Dem Objekt kann über das Parameter `ignore_index` mitgeteilt werden, welches Index automatisch bei der Loss-Berechnung nicht zu berücksichtigen ist. Normalerweise ist das das Padding-Idex. Dadurch kann ein separates Masking entfallen.

Was noch abgedeckt werden könnte:
- Attention
- Bidirektionale LSTM oder Versuch, Input-Sequenzen umzudrehen ("I want to read a book" --> "book a read to want I")
- ...

## 4. Exemplarische Ergebnisse

## 5. Quellen

Papers/Bücher/Online-Beiträge:
- ...
- ...

Meist verwendete Code-Quellen sind:

- PyTorch Chatbot Tutorial, Chatbot Tutorial: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html 
- Machine Learning Mastery, How to Develop a Neural Machine Translation System from Scratch: https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/

Weitere Angaben befinden sich im Code.
