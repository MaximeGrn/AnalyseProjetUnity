# Rapport d'analyse - Comparaison des 4 agencements
*Généré le 2026-01-19 19:46:11*

## Contexte
Ce rapport compare les performances de 4 agencements de magasin (scènes) basés sur des simulations Unity.

## Contrôles de comparabilité
✅ **Toutes les conditions de comparabilité sont remplies :**
- Seed identique entre toutes les scènes
- UtiliserSeedFixe = 1 partout
- Nombre de clients identique
- Taux de completion identique

## Tableau des KPI clés

| Catégorie | Indicateur | Scene_1 | Scene_2 | Scene_3 | Scene_4 |
|---|---|---|---|---|---|
| Global | DureeTotaleSecondes | 122.50 | 128.64 | 141.99 | 132.90 |
| Global | VolumeVentes | 299.00 | 287.00 | 301.00 | 260.00 |
| Performance | DebitClients | 14.69 | 13.99 | 12.68 | 13.54 |
| Performance | PanierMoyen | 9.97 | 9.57 | 10.03 | 8.67 |
| Parcours | IndiceDetour | 1.40 | 1.17 | 1.23 | 1.41 |
| Parcours | TotalRepassages | 71.00 | 63.00 | 70.00 | 73.00 |
| Parcours | TauxRepassageGlobal | 236.70 | 210.00 | 233.30 | 243.30 |
| Parcours | DistanceTotaleParcourue | 3679.88 | 3445.84 | 3260.52 | 3706.74 |
| Parcours | DistanceTotaleOptimale | 2632.15 | 2942.10 | 2661.16 | 2637.49 |
| Layout | IndiceFrustrationAttente | 11.30 | 16.30 | 9.10 | 11.80 |
| Caisses | AttenteMoyenneSecondes | 5.25 | 7.65 | 3.68 | 5.17 |
| Caisses | Saturations | 15.00 | 24.00 | 0.00 | 13.00 |
| Achat | TauxImpulsif | 12.00 | 11.10 | 17.90 | 12.30 |
| Layout | DensiteAchat | 0.08 | 0.08 | 0.09 | 0.07 |
| Temps | TempsMoyenShoppingParClient | 46.62 | 46.98 | 40.52 | 43.89 |
| Parcours | ZoneConfusePire | arme | arme | arme | arme |
| Parcours | ZoneConfusePireRetours | 26.00 | 30.00 | 29.00 | 29.00 |
| Derived | EcartDistance | 1047.73 | 503.74 | 599.36 | 1069.25 |
| Derived | RepassagesParClient | 2.37 | 2.10 | 2.33 | 2.43 |
| Derived | VitesseMoyenne | 30.04 | 26.79 | 22.96 | 27.89 |

## Métriques dérivées

| Indicateur | Scene_1 | Scene_2 | Scene_3 | Scene_4 |
|---|---|---|---|---|
| EcartDistance | 1047.73 | 503.74 | 599.36 | 1069.25 |
| RepassagesParClient | 2.37 | 2.10 | 2.33 | 2.43 |
| VitesseMoyenne | 30.04 | 26.79 | 22.96 | 27.89 |

## Score multi-objectif

| Scène | Score |
| --- | --- |
| Scene_1 | 0.611 |
| Scene_2 | 0.480 |
| Scene_3 | 0.965 |
| Scene_4 | 0.149 |

**Classement :**
1. Scene_3 (score: 0.965)
2. Scene_1 (score: 0.611)
3. Scene_2 (score: 0.480)
4. Scene_4 (score: 0.149)

## Conclusions automatiques

### ⚡ Scène la plus rapide
**Scene_1** avec 122.50 secondes

### 🗺️ Meilleur parcours
**Scene_2** (indice detour + écart distance + repassages minimisés)

### 😌 Meilleur confort
**Scene_3** (saturations + attente + frustration minimisées)

### 💰 Meilleur business
**Scene_3** (ventes + panier + taux impulsif maximisés)

## Recommandation finale

Selon l'objectif recherché :

- **Rapidité** : Privilégier la scène avec la durée totale la plus faible
- **Efficacité du parcours** : Privilégier la scène avec indice de détour minimal
- **Confort client** : Privilégier la scène avec saturations et attentes minimales
- **Performance commerciale** : Privilégier la scène avec ventes et panier moyen maximaux

## Limites et recommandations

⚠️ **Important** : Cette analyse est basée sur une seule simulation par scène.

Pour des statistiques robustes, il est recommandé de :
- Effectuer **N runs** (ex: 30-100) par scène avec des seeds différents
- Calculer des intervalles de confiance pour chaque métrique
- Effectuer des tests statistiques (ANOVA, tests de comparaison de moyennes)
- Analyser la variance et la stabilité des résultats

