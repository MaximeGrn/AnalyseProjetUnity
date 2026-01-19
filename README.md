# Analyseur de Simulations Unity

Script Python pour analyser et comparer 4 exports CSV de simulations Unity (4 agencements de magasin).

## Dépendances

Le script nécessite uniquement les bibliothèques standards suivantes :
- `pandas`
- `numpy`
- `matplotlib`

Installation :
```bash
pip install pandas numpy matplotlib
```

## Utilisation

### Option 1 : Script shell (recommandé)
```bash
./run_analysis.sh
```

### Option 2 : Python directement
```bash
python3.12 analyze_simulations.py
```

ou

```bash
python3 analyze_simulations.py
```

**Note** : Si vous avez plusieurs versions de Python installées, utilisez `python3.12` pour garantir la compatibilité avec les dépendances installées.

Le script va automatiquement :
1. Charger les 4 fichiers CSV depuis `Data/`
2. Vérifier la comparabilité des simulations
3. Générer les tableaux de synthèse
4. Calculer les métriques dérivées
5. Analyser les zones (conversion & ventes)
6. Générer tous les graphiques (PNG)
7. Créer le rapport Markdown
8. Générer la page HTML de visualisation

## Fichiers générés

Tous les fichiers sont créés dans le dossier `outputs/` :

- `summary_wide.csv` - Tableau complet au format large
- `kpi_key.csv` - KPI clés et métriques dérivées
- `report.md` - Rapport détaillé avec conclusions automatiques
- `index.html` - Interface web pour visualiser les résultats
- `zones_conversion.csv` - Données de conversion par zone
- `zones_ventes.csv` - Données de ventes par zone
- `figures/` - Dossier contenant tous les graphiques PNG

## Configuration

Toute la configuration se trouve en haut du script `analyze_simulations.py` dans la section `CONFIG` :
- Chemins des fichiers CSV
- Labels des scènes
- Liste des KPI clés
- Zones à analyser
- Pondérations pour le score multi-objectif (optionnel)
- Port du serveur web (optionnel)

## Visualisation

Après l'exécution, ouvrez `outputs/index.html` dans votre navigateur pour voir tous les résultats de manière interactive.
