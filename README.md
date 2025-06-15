## Description

Cette application Streamlit permet de charger et d'analyser des documents (PDF) en utilisant des frameworks d'indexation basés sur l'IA (LangChain ou LlamaIndex).  
L'utilisateur peut poser des questions sur les documents chargés, recevoir des réponses contextualisées, et donner un feedback sur la qualité des réponses.  
Les feedbacks sont stockés localement dans une base SQLite, et peuvent être exportés au format CSV ou Excel pour analyse.

---

## Fonctionnalités principales

- Chargement et gestion de plusieurs fichiers PDF.
- Choix dynamique du framework d’indexation (LangChain ou LlamaIndex).
- Support multilingue pour les réponses (Français, Anglais, Espagnol, Allemand).
- Pose de questions et réception de réponses générées par l’IA.
- Collecte et stockage des feedbacks utilisateurs (qualité des réponses).
- Interface web simple, interactive et réactive grâce à Streamlit.

---

## Environnement et Prérequis

- Python 3.8 ou supérieur
- Streamlit
- PyYAML
- pandas
- sqlite3 (inclus avec Python)
- Modules spécifiques au projet (`rag.langchain`, `rag.llamaindex`) à installer ou développer selon ton projet
- Système compatible : Windows, macOS, Linux

---
