# Reconnaissance ASL (YOLOv8)

Implémentation de YOLOv8 pour la reconnaissance de la langue des signes américaine (ASL) dans le cadre du Projet de Programmation 2 (HAI606I).

## Installation

1) Installer le paquet [Tkinter](https://docs.python.org/3/library/tkinter.html) : python3-tk
2) (Optionnel) Installer et créer un environnement virtuel avec virtualenv :

```bash
python3 -m venv .
source bin/activate
```

3) Installer les dépendances restantes avec [pip](https://pip.pypa.io/en/stable/) :

```bash
pip install customtkinter
pip install ultralytics
```

## Utilisation

**main.py** est le programme principal pour l'inférence en temps réel avec une caméra ou des images statiques.

**IoU_comparison_video.py** est le script de génération de video pour créer une comparaison de performances entre différents modèles.
