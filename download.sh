#!/usr/bin/env bash


echo "Downloading files..."
wget -N https://www.manythings.org/anki/deu-eng.zip -P ./data/

echo "Unzipping files...."
unzip -o ./data/deu-eng.zip -d ./data/

echo "Removing zip file..."
rm ./data/deu-eng.zip