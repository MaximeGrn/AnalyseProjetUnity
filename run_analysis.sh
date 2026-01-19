#!/bin/bash
# Script de lancement pour l'analyseur de simulations Unity

# Essayer python3.12 d'abord, puis python3 en fallback
if command -v python3.12 &> /dev/null; then
    python3.12 analyze_simulations.py
elif command -v python3 &> /dev/null; then
    python3 analyze_simulations.py
else
    echo "❌ Erreur: Python 3 n'est pas installé ou trouvé"
    exit 1
fi
