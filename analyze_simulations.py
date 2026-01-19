#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyseur de simulations Unity - Comparaison de 4 agencements de magasin
Génère automatiquement tableaux, statistiques, graphiques et rapport.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import webbrowser
import http.server
import socketserver
import threading

# Configuration matplotlib pour éviter les problèmes d'affichage
matplotlib.use('Agg')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chemins des fichiers CSV d'entrée
CSV_PATHS = [
    'Data/Rapport_Simulation_Scene_1.csv',
    'Data/Rapport_Simulation_Scene_2.csv',
    'Data/Rapport_Simulation_Scene_3.csv',
    'Data/Rapport_Simulation_Scene_4.csv'
]

# Labels des scènes (pour les graphiques et le rapport)
SCENE_LABELS = ['Scene_1', 'Scene_2', 'Scene_3', 'Scene_4']

# Dossier de sortie
OUTPUT_DIR = 'outputs'
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# Liste des KPI clés à extraire
KPI_KEY_LIST = [
    'Global::DureeTotaleSecondes',
    'Global::VolumeVentes',
    'Performance::DebitClients',
    'Performance::PanierMoyen',
    'Parcours::IndiceDetour',
    'Parcours::TotalRepassages',
    'Parcours::TauxRepassageGlobal',
    'Parcours::DistanceTotaleParcourue',
    'Parcours::DistanceTotaleOptimale',
    'Layout::IndiceFrustrationAttente',
    'Caisses::AttenteMoyenneSecondes',
    'Caisses::Saturations',
    'Achat::TauxImpulsif',
    'Layout::DensiteAchat',
    'Temps::TempsMoyenShoppingParClient',
    'Parcours::ZoneConfusePire',
    'Parcours::ZoneConfusePireRetours'
]

# Zones à analyser
ZONES = ['alcool', 'arme', 'chips', 'lait', 'pain', 'soft', 'viande']

# Pondérations pour le score multi-objectif (optionnel)
# Format: {kpi: poids} où poids > 0 signifie "plus c'est mieux", < 0 signifie "moins c'est mieux"
SCORE_WEIGHTS = {
    'Global::VolumeVentes': 0.3,
    'Performance::PanierMoyen': 0.2,
    'Achat::TauxImpulsif': 0.1,
    'Parcours::IndiceDetour': -0.15,  # négatif = moins c'est mieux
    'Caisses::Saturations': -0.15,  # négatif = moins c'est mieux
    'Caisses::AttenteMoyenneSecondes': -0.1,  # négatif = moins c'est mieux
}

# Port pour le serveur web local (optionnel)
WEB_SERVER_PORT = 8000
AUTO_OPEN_BROWSER = True  # Ouvrir automatiquement le navigateur


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def parse_duration(duration_str: str) -> Optional[float]:
    """
    Parse une durée formatée "mm:ss" ou "01m:47s" en secondes.
    Retourne None si le format n'est pas reconnu.
    """
    if pd.isna(duration_str) or not isinstance(duration_str, str):
        return None
    
    # Format "mm:ss" ou "01m:47s"
    pattern1 = r'(\d+)m:(\d+)s'
    pattern2 = r'(\d+):(\d+)'
    
    match = re.search(pattern1, duration_str)
    if match:
        minutes, seconds = int(match.group(1)), int(match.group(2))
        return minutes * 60 + seconds
    
    match = re.search(pattern2, duration_str)
    if match:
        minutes, seconds = int(match.group(1)), int(match.group(2))
        return minutes * 60 + seconds
    
    return None


def parse_numeric(value: str) -> Tuple[Optional[float], str]:
    """
    Essaie de convertir une valeur en float.
    Gère les virgules comme séparateurs décimaux.
    Retourne (valeur_numérique, valeur_string_originale).
    """
    if pd.isna(value):
        return None, str(value)
    
    value_str = str(value).strip()
    
    # Essayer de parser une durée d'abord
    duration_sec = parse_duration(value_str)
    if duration_sec is not None:
        return duration_sec, value_str
    
    # Remplacer virgule par point pour conversion numérique
    value_clean = value_str.replace(',', '.')
    
    try:
        return float(value_clean), value_str
    except (ValueError, TypeError):
        return None, value_str


# ============================================================================
# CHARGEMENT ET NETTOYAGE
# ============================================================================

def load_and_clean(csv_path: str, scene_label: str) -> pd.DataFrame:
    """
    Charge un CSV et nettoie les données.
    Retourne un DataFrame avec colonnes: CATEGORIE, INDICATEUR, VALEUR_NUM, VALEUR_STR, UNITE
    """
    print(f"  Chargement: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")
    
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    
    # Vérifier les colonnes attendues
    required_cols = ['CATEGORIE', 'INDICATEUR', 'VALEUR', 'UNITE']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path}: {missing}")
    
    # Parser les valeurs
    df['VALEUR_NUM'] = None
    df['VALEUR_STR'] = None
    
    for idx, row in df.iterrows():
        num_val, str_val = parse_numeric(row['VALEUR'])
        df.at[idx, 'VALEUR_NUM'] = num_val
        df.at[idx, 'VALEUR_STR'] = str_val
    
    df['SCENE'] = scene_label
    
    return df


def check_comparability(dfs: List[pd.DataFrame], scene_labels: List[str]) -> Dict[str, any]:
    """
    Vérifie la comparabilité entre les scènes.
    Retourne un dictionnaire avec les résultats des vérifications.
    """
    checks = {
        'seed_identique': True,
        'seed_fixe': True,
        'clients_identiques': True,
        'completion_identique': True,
        'warnings': []
    }
    
    # Extraire les valeurs de contrôle
    seeds = []
    seed_fixes = []
    clients = []
    completions = []
    
    for df, label in zip(dfs, scene_labels):
        seed_val = df[(df['CATEGORIE'] == 'Config') & (df['INDICATEUR'] == 'Seed')]['VALEUR_NUM'].values
        seed_fix_val = df[(df['CATEGORIE'] == 'Config') & (df['INDICATEUR'] == 'UtiliserSeedFixe')]['VALEUR_NUM'].values
        clients_val = df[(df['CATEGORIE'] == 'Global') & (df['INDICATEUR'] == 'ClientsTraites')]['VALEUR_NUM'].values
        completion_val = df[(df['CATEGORIE'] == 'Global') & (df['INDICATEUR'] == 'TauxCompletion')]['VALEUR_NUM'].values
        
        if len(seed_val) > 0:
            seeds.append(seed_val[0])
        if len(seed_fix_val) > 0:
            seed_fixes.append(seed_fix_val[0])
        if len(clients_val) > 0:
            clients.append(clients_val[0])
        if len(completion_val) > 0:
            completions.append(completion_val[0])
    
    # Vérifications
    if len(set(seeds)) > 1:
        checks['seed_identique'] = False
        checks['warnings'].append(f"⚠️ Seeds différents: {seeds}")
    
    if not all(sf == 1 for sf in seed_fixes):
        checks['seed_fixe'] = False
        checks['warnings'].append(f"⚠️ UtiliserSeedFixe n'est pas 1 partout: {seed_fixes}")
    
    if len(set(clients)) > 1:
        checks['clients_identiques'] = False
        checks['warnings'].append(f"⚠️ Nombre de clients différents: {clients}")
    
    if len(set(completions)) > 1:
        checks['completion_identique'] = False
        checks['warnings'].append(f"⚠️ Taux de completion différents: {completions}")
    
    return checks


# ============================================================================
# RESTRUCTURATION DES DONNÉES
# ============================================================================

def build_wide_table(dfs: List[pd.DataFrame], scene_labels: List[str]) -> pd.DataFrame:
    """
    Construit une table "wide" avec index (CATEGORIE, INDICATEUR) et colonnes Scene_1..Scene_4.
    Utilise VALEUR_NUM quand disponible, sinon VALEUR_STR.
    """
    all_data = []
    
    for df, label in zip(dfs, scene_labels):
        for _, row in df.iterrows():
            key = (row['CATEGORIE'], row['INDICATEUR'])
            value = row['VALEUR_NUM'] if row['VALEUR_NUM'] is not None else row['VALEUR_STR']
            all_data.append({
                'CATEGORIE': key[0],
                'INDICATEUR': key[1],
                'SCENE': label,
                'VALEUR': value
            })
    
    df_long = pd.DataFrame(all_data)
    df_wide = df_long.pivot_table(
        index=['CATEGORIE', 'INDICATEUR'],
        columns='SCENE',
        values='VALEUR',
        aggfunc='first'
    )
    
    return df_wide


def extract_kpi_table(df_wide: pd.DataFrame, kpi_list: List[str]) -> pd.DataFrame:
    """
    Extrait les KPI clés dans une table dédiée.
    """
    kpi_data = []
    
    for kpi in kpi_list:
        cat, ind = kpi.split('::')
        if (cat, ind) in df_wide.index:
            row = df_wide.loc[(cat, ind)]
            kpi_data.append({
                'CATEGORIE': cat,
                'INDICATEUR': ind,
                **{col: row[col] for col in df_wide.columns}
            })
    
    return pd.DataFrame(kpi_data)


# ============================================================================
# MÉTRIQUES DÉRIVÉES
# ============================================================================

def compute_derived_metrics(df_kpi: pd.DataFrame, df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les métriques dérivées au tableau KPI.
    """
    df_derived = df_kpi.copy()
    
    # Extraire Global::ClientsTraites pour les calculs
    clients_traites = {}
    for scene in df_wide.columns:
        cat, ind = 'Global', 'ClientsTraites'
        if (cat, ind) in df_wide.index:
            val = df_wide.loc[(cat, ind), scene]
            if isinstance(val, (int, float)) and not pd.isna(val):
                clients_traites[scene] = val
    
    # Calculer les métriques dérivées pour chaque scène
    # Structure: {indicateur: {scene: valeur}}
    derived_data = {
        'EcartDistance': {},
        'RepassagesParClient': {},
        'VitesseMoyenne': {}
    }
    
    for scene in df_wide.columns:
        # EcartDistance
        dist_parc = None
        dist_opt = None
        if ('Parcours', 'DistanceTotaleParcourue') in df_wide.index:
            dist_parc = df_wide.loc[('Parcours', 'DistanceTotaleParcourue'), scene]
        if ('Parcours', 'DistanceTotaleOptimale') in df_wide.index:
            dist_opt = df_wide.loc[('Parcours', 'DistanceTotaleOptimale'), scene]
        
        if isinstance(dist_parc, (int, float)) and isinstance(dist_opt, (int, float)):
            ecart = dist_parc - dist_opt
            derived_data['EcartDistance'][scene] = ecart
        
        # RepassagesParClient
        total_repassages = None
        if ('Parcours', 'TotalRepassages') in df_wide.index:
            total_repassages = df_wide.loc[('Parcours', 'TotalRepassages'), scene]
        
        if isinstance(total_repassages, (int, float)) and scene in clients_traites and clients_traites[scene] > 0:
            repassages_pc = total_repassages / clients_traites[scene]
            derived_data['RepassagesParClient'][scene] = repassages_pc
        
        # VitesseMoyenne
        duree_totale = None
        if ('Global', 'DureeTotaleSecondes') in df_wide.index:
            duree_totale = df_wide.loc[('Global', 'DureeTotaleSecondes'), scene]
        
        if isinstance(dist_parc, (int, float)) and isinstance(duree_totale, (int, float)) and duree_totale > 0:
            vitesse = dist_parc / duree_totale
            derived_data['VitesseMoyenne'][scene] = vitesse
    
    # Ajouter les lignes dérivées
    for indicateur, scene_values in derived_data.items():
        if scene_values:
            new_row = {'CATEGORIE': 'Derived', 'INDICATEUR': indicateur}
            for scene in df_wide.columns:
                new_row[scene] = scene_values.get(scene, None)
            df_derived = pd.concat([df_derived, pd.DataFrame([new_row])], ignore_index=True)
    
    return df_derived


def compute_scores(df_kpi: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Calcule un score multi-objectif normalisé (min-max) avec pondérations.
    Retourne un DataFrame avec les scores par scène.
    """
    if not weights:
        return None
    
    scores = {}
    
    for scene in df_kpi.columns:
        if scene in ['CATEGORIE', 'INDICATEUR']:
            continue
        
        scene_scores = []
        
        for kpi, weight in weights.items():
            cat, ind = kpi.split('::')
            row = df_kpi[(df_kpi['CATEGORIE'] == cat) & (df_kpi['INDICATEUR'] == ind)]
            
            if len(row) > 0:
                values = []
                for s in df_kpi.columns:
                    if s not in ['CATEGORIE', 'INDICATEUR']:
                        val = row[s].values[0]
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            values.append(val)
                
                if len(values) > 1:
                    val_scene = row[scene].values[0]
                    if isinstance(val_scene, (int, float)) and not pd.isna(val_scene):
                        min_val, max_val = min(values), max(values)
                        if max_val > min_val:
                            normalized = (val_scene - min_val) / (max_val - min_val)
                        else:
                            normalized = 0.5
                        
                        # Appliquer le poids (négatif = moins c'est mieux, donc inverser)
                        if weight < 0:
                            normalized = 1 - normalized
                            weight = abs(weight)
                        
                        scene_scores.append(normalized * weight)
        
        scores[scene] = sum(scene_scores) if scene_scores else None
    
    if scores:
        df_scores = pd.DataFrame([{
            'CATEGORIE': 'Score',
            'INDICATEUR': 'ScoreMultiObjectif',
            **scores
        }])
        return df_scores
    
    return None


# ============================================================================
# ANALYSE DES ZONES
# ============================================================================

def extract_zone_data(df_wide: pd.DataFrame, zones: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Extrait les données de conversion et ventes par zone.
    Retourne un dictionnaire avec 'conversion' et 'ventes'.
    """
    conversion_data = []
    ventes_data = []
    
    for zone in zones:
        zone_row_conv = {'Zone': zone}
        zone_row_ventes = {'Zone': zone}
        
        for scene in df_wide.columns:
            # Conversion
            visites_key = ('Conversion', f'{zone}_Visites')
            achats_key = ('Conversion', f'{zone}_Achats')
            taux_key = ('Conversion', f'{zone}_Taux')
            
            visites = df_wide.loc[visites_key, scene] if visites_key in df_wide.index else None
            achats = df_wide.loc[achats_key, scene] if achats_key in df_wide.index else None
            taux = df_wide.loc[taux_key, scene] if taux_key in df_wide.index else None
            
            zone_row_conv[f'{scene}_Visites'] = visites
            zone_row_conv[f'{scene}_Achats'] = achats
            zone_row_conv[f'{scene}_Taux'] = taux
            
            # Ventes
            ventes_key = ('VentesParZone', zone)
            ventes = df_wide.loc[ventes_key, scene] if ventes_key in df_wide.index else None
            zone_row_ventes[scene] = ventes
        
        conversion_data.append(zone_row_conv)
        ventes_data.append(zone_row_ventes)
    
    return {
        'conversion': pd.DataFrame(conversion_data),
        'ventes': pd.DataFrame(ventes_data)
    }


# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_kpi_comparison(df_kpi: pd.DataFrame, kpi_name: str, output_path: str):
    """
    Génère un bar chart comparatif pour un KPI donné.
    """
    cat, ind = kpi_name.split('::')
    row = df_kpi[(df_kpi['CATEGORIE'] == cat) & (df_kpi['INDICATEUR'] == ind)]
    
    if len(row) == 0:
        print(f"    ⚠️ KPI non trouvé: {kpi_name}")
        return
    
    scenes = [c for c in df_kpi.columns if c not in ['CATEGORIE', 'INDICATEUR']]
    values = [row[s].values[0] for s in scenes]
    
    # Filtrer les valeurs non numériques
    numeric_data = []
    numeric_scenes = []
    for s, v in zip(scenes, values):
        if isinstance(v, (int, float)) and not pd.isna(v):
            numeric_data.append(v)
            numeric_scenes.append(s)
    
    if not numeric_data:
        print(f"    ⚠️ Aucune valeur numérique pour {kpi_name}")
        return
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(numeric_scenes, numeric_data, color='steelblue', alpha=0.7)
    plt.title(f'{cat} - {ind}', fontsize=14, fontweight='bold')
    plt.xlabel('Scène', fontsize=12)
    plt.ylabel('Valeur', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {os.path.basename(output_path)}")


def plot_scatter(df_kpi: pd.DataFrame, x_kpi: str, y_kpi: str, output_path: str):
    """
    Génère un scatter plot comparant deux KPI.
    """
    x_cat, x_ind = x_kpi.split('::')
    y_cat, y_ind = y_kpi.split('::')
    
    x_row = df_kpi[(df_kpi['CATEGORIE'] == x_cat) & (df_kpi['INDICATEUR'] == x_ind)]
    y_row = df_kpi[(df_kpi['CATEGORIE'] == y_cat) & (df_kpi['INDICATEUR'] == y_ind)]
    
    if len(x_row) == 0 or len(y_row) == 0:
        print(f"    ⚠️ KPI non trouvé pour scatter: {x_kpi} vs {y_kpi}")
        return
    
    scenes = [c for c in df_kpi.columns if c not in ['CATEGORIE', 'INDICATEUR']]
    x_values = []
    y_values = []
    scene_labels_clean = []
    
    for s in scenes:
        x_val = x_row[s].values[0]
        y_val = y_row[s].values[0]
        if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
            if not pd.isna(x_val) and not pd.isna(y_val):
                x_values.append(x_val)
                y_values.append(y_val)
                scene_labels_clean.append(s)
    
    if len(x_values) < 2:
        print(f"    ⚠️ Pas assez de données pour scatter: {x_kpi} vs {y_kpi}")
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, s=200, alpha=0.6, c='coral', edgecolors='black', linewidth=2)
    
    for i, label in enumerate(scene_labels_clean):
        plt.annotate(label, (x_values[i], y_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=11)
    
    plt.xlabel(f'{x_cat} - {x_ind}', fontsize=12)
    plt.ylabel(f'{y_cat} - {y_ind}', fontsize=12)
    plt.title(f'{x_ind} vs {y_ind}', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {os.path.basename(output_path)}")


def plot_zone_heatmap(df_zone_conv: pd.DataFrame, output_path: str):
    """
    Génère une heatmap des taux de conversion par zone.
    """
    scenes = [c for c in df_zone_conv.columns if c.endswith('_Taux')]
    zones = df_zone_conv['Zone'].values
    
    if not scenes:
        print("    ⚠️ Aucune donnée de conversion trouvée")
        return
    
    # Construire la matrice
    data_matrix = []
    scene_labels_clean = [s.replace('_Taux', '') for s in scenes]
    
    for zone in zones:
        row = df_zone_conv[df_zone_conv['Zone'] == zone]
        zone_values = []
        for s in scenes:
            val = row[s].values[0]
            if isinstance(val, (int, float)) and not pd.isna(val):
                zone_values.append(val)
            else:
                zone_values.append(0)
        data_matrix.append(zone_values)
    
    data_matrix = np.array(data_matrix)
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Taux de conversion (%)')
    
    plt.xticks(range(len(scene_labels_clean)), scene_labels_clean, rotation=0)
    plt.yticks(range(len(zones)), zones)
    plt.xlabel('Scène', fontsize=12)
    plt.ylabel('Zone', fontsize=12)
    plt.title('Taux de conversion par zone', fontsize=14, fontweight='bold')
    
    # Ajouter les valeurs dans les cellules
    for i in range(len(zones)):
        for j in range(len(scene_labels_clean)):
            text = plt.text(j, i, f'{data_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {os.path.basename(output_path)}")


def plot_zone_sales(df_zone_ventes: pd.DataFrame, output_path: str):
    """
    Génère un graphique comparatif des ventes par zone (bar chart empilé et groupé).
    """
    scenes = [c for c in df_zone_ventes.columns if c != 'Zone']
    zones = df_zone_ventes['Zone'].values
    
    if not scenes:
        print("    ⚠️ Aucune donnée de ventes trouvée")
        return
    
    # Préparer les données
    data_dict = {}
    for scene in scenes:
        data_dict[scene] = []
        for zone in zones:
            row = df_zone_ventes[df_zone_ventes['Zone'] == zone]
            val = row[scene].values[0]
            if isinstance(val, (int, float)) and not pd.isna(val):
                data_dict[scene].append(val)
            else:
                data_dict[scene].append(0)
    
    # Graphique empilé
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(zones))
    width = 0.2
    colors = plt.cm.Set3(np.linspace(0, 1, len(scenes)))
    
    # Empilé
    bottom = np.zeros(len(zones))
    for i, scene in enumerate(scenes):
        ax1.bar(x, data_dict[scene], width, bottom=bottom, label=scene, color=colors[i])
        bottom += data_dict[scene]
    
    ax1.set_xlabel('Zone', fontsize=12)
    ax1.set_ylabel('Ventes', fontsize=12)
    ax1.set_title('Ventes par zone (empilé)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(zones, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Groupé
    for i, scene in enumerate(scenes):
        ax2.bar(x + i * width, data_dict[scene], width, label=scene, color=colors[i])
    
    ax2.set_xlabel('Zone', fontsize=12)
    ax2.set_ylabel('Ventes', fontsize=12)
    ax2.set_title('Ventes par zone (groupé)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width * (len(scenes) - 1) / 2)
    ax2.set_xticklabels(zones, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {os.path.basename(output_path)}")


# ============================================================================
# RAPPORT AUTOMATIQUE
# ============================================================================

def generate_report(df_kpi: pd.DataFrame, df_wide: pd.DataFrame, 
                   checks: Dict, zone_data: Dict[str, pd.DataFrame],
                   df_scores: Optional[pd.DataFrame], output_path: str):
    """
    Génère un rapport Markdown automatique.
    """
    report_lines = []
    
    report_lines.append("# Rapport d'analyse - Comparaison des 4 agencements\n")
    report_lines.append(f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    
    # Contexte
    report_lines.append("## Contexte\n")
    report_lines.append("Ce rapport compare les performances de 4 agencements de magasin ")
    report_lines.append("(scènes) basés sur des simulations Unity.\n\n")
    
    # Contrôles de comparabilité
    report_lines.append("## Contrôles de comparabilité\n")
    if all([checks['seed_identique'], checks['seed_fixe'], 
            checks['clients_identiques'], checks['completion_identique']]):
        report_lines.append("✅ **Toutes les conditions de comparabilité sont remplies :**\n")
        report_lines.append("- Seed identique entre toutes les scènes\n")
        report_lines.append("- UtiliserSeedFixe = 1 partout\n")
        report_lines.append("- Nombre de clients identique\n")
        report_lines.append("- Taux de completion identique\n\n")
    else:
        report_lines.append("⚠️ **Attention : certaines conditions de comparabilité ne sont pas remplies :**\n\n")
        for warning in checks['warnings']:
            report_lines.append(f"- {warning}\n")
        report_lines.append("\n")
    
    # Tableau KPI
    report_lines.append("## Tableau des KPI clés\n\n")
    report_lines.append("| Catégorie | Indicateur | " + " | ".join([c for c in df_kpi.columns if c not in ['CATEGORIE', 'INDICATEUR']]) + " |\n")
    report_lines.append("|" + "---|" * (len(df_kpi.columns)) + "\n")
    
    for _, row in df_kpi.iterrows():
        cat = str(row['CATEGORIE'])
        ind = str(row['INDICATEUR'])
        values = []
        for col in df_kpi.columns:
            if col not in ['CATEGORIE', 'INDICATEUR']:
                val = row[col]
                if isinstance(val, (int, float)) and not pd.isna(val):
                    values.append(f"{val:.2f}")
                else:
                    values.append(str(val))
        report_lines.append(f"| {cat} | {ind} | " + " | ".join(values) + " |\n")
    
    report_lines.append("\n")
    
    # Métriques dérivées
    derived_rows = df_kpi[df_kpi['CATEGORIE'] == 'Derived']
    if len(derived_rows) > 0:
        report_lines.append("## Métriques dérivées\n\n")
        report_lines.append("| Indicateur | " + " | ".join([c for c in df_kpi.columns if c not in ['CATEGORIE', 'INDICATEUR']]) + " |\n")
        report_lines.append("|" + "---|" * (len(df_kpi.columns) - 1) + "\n")
        
        for _, row in derived_rows.iterrows():
            ind = str(row['INDICATEUR'])
            values = []
            for col in df_kpi.columns:
                if col not in ['CATEGORIE', 'INDICATEUR']:
                    val = row[col]
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        values.append(f"{val:.2f}")
                    else:
                        values.append(str(val))
            report_lines.append(f"| {ind} | " + " | ".join(values) + " |\n")
        
        report_lines.append("\n")
    
    # Scores
    if df_scores is not None and len(df_scores) > 0:
        report_lines.append("## Score multi-objectif\n\n")
        score_row = df_scores.iloc[0]
        report_lines.append("| Scène | Score |\n")
        report_lines.append("| --- | --- |\n")
        for col in df_scores.columns:
            if col not in ['CATEGORIE', 'INDICATEUR']:
                val = score_row[col]
                if isinstance(val, (int, float)) and not pd.isna(val):
                    report_lines.append(f"| {col} | {val:.3f} |\n")
        
        # Ranking
        score_dict = {col: score_row[col] for col in df_scores.columns 
                     if col not in ['CATEGORIE', 'INDICATEUR'] 
                     and isinstance(score_row[col], (int, float)) and not pd.isna(score_row[col])}
        sorted_scenes = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        report_lines.append("\n**Classement :**\n")
        for i, (scene, score) in enumerate(sorted_scenes, 1):
            report_lines.append(f"{i}. {scene} (score: {score:.3f})\n")
        report_lines.append("\n")
    
    # Conclusions automatiques
    report_lines.append("## Conclusions automatiques\n\n")
    
    scenes = [c for c in df_kpi.columns if c not in ['CATEGORIE', 'INDICATEUR']]
    
    # Scène la plus rapide
    duree_key = ('Global', 'DureeTotaleSecondes')
    if duree_key in df_wide.index:
        durees = {}
        for scene in scenes:
            val = df_wide.loc[duree_key, scene]
            if isinstance(val, (int, float)) and not pd.isna(val):
                durees[scene] = val
        if durees:
            scene_rapide = min(durees.items(), key=lambda x: x[1])[0]
            report_lines.append(f"### ⚡ Scène la plus rapide\n")
            report_lines.append(f"**{scene_rapide}** avec {durees[scene_rapide]:.2f} secondes\n\n")
    
    # Meilleur parcours
    detour_key = ('Parcours', 'IndiceDetour')
    repassages_key = ('Parcours', 'TotalRepassages')
    
    # Récupérer EcartDistance depuis df_kpi (métrique dérivée)
    ecart_row = df_kpi[(df_kpi['CATEGORIE'] == 'Derived') & (df_kpi['INDICATEUR'] == 'EcartDistance')]
    
    parcours_scores = {}
    for scene in scenes:
        score = 0
        if detour_key in df_wide.index:
            val = df_wide.loc[detour_key, scene]
            if isinstance(val, (int, float)) and not pd.isna(val):
                score += val  # Plus bas = mieux
        if len(ecart_row) > 0:
            val = ecart_row[scene].values[0]
            if isinstance(val, (int, float)) and not pd.isna(val):
                score += val / 100  # Normaliser
        if repassages_key in df_wide.index:
            val = df_wide.loc[repassages_key, scene]
            if isinstance(val, (int, float)) and not pd.isna(val):
                score += val / 10  # Normaliser
        
        if score > 0:
            parcours_scores[scene] = score
    
    if parcours_scores:
        scene_parcours = min(parcours_scores.items(), key=lambda x: x[1])[0]
        report_lines.append(f"### 🗺️ Meilleur parcours\n")
        report_lines.append(f"**{scene_parcours}** (indice detour + écart distance + repassages minimisés)\n\n")
    
    # Meilleur confort
    sat_key = ('Caisses', 'Saturations')
    attente_key = ('Caisses', 'AttenteMoyenneSecondes')
    frust_key = ('Layout', 'IndiceFrustrationAttente')
    
    confort_scores = {}
    for scene in scenes:
        score = 0
        if sat_key in df_wide.index:
            val = df_wide.loc[sat_key, scene]
            if isinstance(val, (int, float)) and not pd.isna(val):
                score += val
        if attente_key in df_wide.index:
            val = df_wide.loc[attente_key, scene]
            if isinstance(val, (int, float)) and not pd.isna(val):
                score += val
        if frust_key in df_wide.index:
            val = df_wide.loc[frust_key, scene]
            if isinstance(val, (int, float)) and not pd.isna(val):
                score += val
        
        if score > 0:
            confort_scores[scene] = score
    
    if confort_scores:
        scene_confort = min(confort_scores.items(), key=lambda x: x[1])[0]
        report_lines.append(f"### 😌 Meilleur confort\n")
        report_lines.append(f"**{scene_confort}** (saturations + attente + frustration minimisées)\n\n")
    
    # Meilleur business
    ventes_key = ('Global', 'VolumeVentes')
    panier_key = ('Performance', 'PanierMoyen')
    impulsif_key = ('Achat', 'TauxImpulsif')
    
    business_scores = {}
    for scene in scenes:
        score = 0
        if ventes_key in df_wide.index:
            val = df_wide.loc[ventes_key, scene]
            if isinstance(val, (int, float)) and not pd.isna(val):
                score += val / 10  # Normaliser
        if panier_key in df_wide.index:
            val = df_wide.loc[panier_key, scene]
            if isinstance(val, (int, float)) and not pd.isna(val):
                score += val
        if impulsif_key in df_wide.index:
            val = df_wide.loc[impulsif_key, scene]
            if isinstance(val, (int, float)) and not pd.isna(val):
                score += val
        
        if score > 0:
            business_scores[scene] = score
    
    if business_scores:
        scene_business = max(business_scores.items(), key=lambda x: x[1])[0]
        report_lines.append(f"### 💰 Meilleur business\n")
        report_lines.append(f"**{scene_business}** (ventes + panier + taux impulsif maximisés)\n\n")
    
    # Recommandation
    report_lines.append("## Recommandation finale\n\n")
    report_lines.append("Selon l'objectif recherché :\n\n")
    report_lines.append("- **Rapidité** : Privilégier la scène avec la durée totale la plus faible\n")
    report_lines.append("- **Efficacité du parcours** : Privilégier la scène avec indice de détour minimal\n")
    report_lines.append("- **Confort client** : Privilégier la scène avec saturations et attentes minimales\n")
    report_lines.append("- **Performance commerciale** : Privilégier la scène avec ventes et panier moyen maximaux\n\n")
    
    # Limites
    report_lines.append("## Limites et recommandations\n\n")
    report_lines.append("⚠️ **Important** : Cette analyse est basée sur une seule simulation par scène.\n\n")
    report_lines.append("Pour des statistiques robustes, il est recommandé de :\n")
    report_lines.append("- Effectuer **N runs** (ex: 30-100) par scène avec des seeds différents\n")
    report_lines.append("- Calculer des intervalles de confiance pour chaque métrique\n")
    report_lines.append("- Effectuer des tests statistiques (ANOVA, tests de comparaison de moyennes)\n")
    report_lines.append("- Analyser la variance et la stabilité des résultats\n\n")
    
    # Écrire le rapport
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"  ✓ Rapport généré: {output_path}")


# ============================================================================
# INTERFACE HTML
# ============================================================================

def build_html(df_kpi: pd.DataFrame, output_path: str, figures_dir: str):
    """
    Génère une page HTML simple pour visualiser les résultats.
    """
    html_lines = []
    
    html_lines.append("""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse des Simulations Unity</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        .section {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .figure-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .figure-item {
            text-align: center;
        }
        .figure-item img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .figure-item p {
            margin-top: 10px;
            font-weight: bold;
            color: #666;
        }
        a {
            color: #4CAF50;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .links {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 4px;
        }
        .links ul {
            list-style-type: none;
            padding-left: 0;
        }
        .links li {
            margin: 8px 0;
        }
    </style>
</head>
<body>
    <h1>📊 Analyse des Simulations Unity - Comparaison des 4 Agencements</h1>
    
    <div class="section">
        <h2>📁 Fichiers générés</h2>
        <div class="links">
            <ul>
                <li>📄 <a href="summary_wide.csv">summary_wide.csv</a> - Tableau complet (format large)</li>
                <li>📄 <a href="kpi_key.csv">kpi_key.csv</a> - KPI clés et métriques dérivées</li>
                <li>📄 <a href="report.md">report.md</a> - Rapport détaillé</li>
            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 KPI Clés</h2>
        <div class="figure-grid">
""")
    
    # Lister les figures KPI
    kpi_figures = []
    for kpi in KPI_KEY_LIST:
        cat, ind = kpi.split('::')
        safe_name = f"{cat}_{ind}".replace('::', '_').replace('/', '_')
        fig_path = os.path.join('figures', f'kpi_{safe_name}.png')
        if os.path.exists(os.path.join(output_path.replace('index.html', ''), fig_path)):
            kpi_figures.append((fig_path, f"{cat} - {ind}"))
    
    for fig_path, title in kpi_figures:
        html_lines.append(f"""            <div class="figure-item">
                <img src="{fig_path}" alt="{title}">
                <p>{title}</p>
            </div>
""")
    
    html_lines.append("""        </div>
    </div>
    
    <div class="section">
        <h2>🗺️ Analyse des Zones</h2>
        <div class="figure-grid">
""")
    
    # Figures zones
    zone_figures = [
        ('figures/zone_conversion_heatmap.png', 'Taux de conversion par zone'),
        ('figures/zone_ventes_comparison.png', 'Ventes par zone')
    ]
    
    for fig_path, title in zone_figures:
        full_path = os.path.join(output_path.replace('index.html', ''), fig_path)
        if os.path.exists(full_path):
            html_lines.append(f"""            <div class="figure-item">
                <img src="{fig_path}" alt="{title}">
                <p>{title}</p>
            </div>
""")
    
    html_lines.append("""        </div>
    </div>
    
    <div class="section">
        <h2>📊 Analyses comparatives (Scatter plots)</h2>
        <div class="figure-grid">
""")
    
    # Scatter plots
    scatter_figures = [
        ('figures/scatter_IndiceDetour_vs_VolumeVentes.png', 'IndiceDetour vs VolumeVentes'),
        ('figures/scatter_Saturations_vs_DebitClients.png', 'Saturations vs DebitClients'),
        ('figures/scatter_AttenteMoyenneSecondes_vs_IndiceFrustrationAttente.png', 'Attente vs Frustration')
    ]
    
    for fig_path, title in scatter_figures:
        full_path = os.path.join(output_path.replace('index.html', ''), fig_path)
        if os.path.exists(full_path):
            html_lines.append(f"""            <div class="figure-item">
                <img src="{fig_path}" alt="{title}">
                <p>{title}</p>
            </div>
""")
    
    html_lines.append("""        </div>
    </div>
    
    <div class="section">
        <p><em>Généré automatiquement par analyze_simulations.py</em></p>
    </div>
</body>
</html>""")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(html_lines)
    
    print(f"  ✓ Page HTML générée: {output_path}")


def start_web_server(port: int, directory: str):
    """
    Démarre un serveur web local pour servir les fichiers.
    """
    os.chdir(directory)
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)
    print(f"\n🌐 Serveur web démarré sur http://localhost:{port}/")
    print(f"   Ouvrez votre navigateur à cette adresse pour voir les résultats.")
    print(f"   Appuyez sur Ctrl+C pour arrêter le serveur.\n")
    httpd.serve_forever()


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale : orchestre toute l'analyse.
    """
    print("=" * 70)
    print("ANALYSEUR DE SIMULATIONS UNITY")
    print("=" * 70)
    print()
    
    # Créer les dossiers de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"📁 Dossiers créés: {OUTPUT_DIR}/, {FIGURES_DIR}/")
    print()
    
    # 1. Chargement et nettoyage
    print("📥 Étape 1: Chargement et nettoyage des données...")
    dfs = []
    for csv_path, label in zip(CSV_PATHS, SCENE_LABELS):
        try:
            df = load_and_clean(csv_path, label)
            dfs.append(df)
        except Exception as e:
            print(f"  ❌ Erreur lors du chargement de {csv_path}: {e}")
            return
    
    print(f"  ✓ {len(dfs)} fichiers chargés avec succès\n")
    
    # 2. Vérification de comparabilité
    print("🔍 Étape 2: Vérification de comparabilité...")
    checks = check_comparability(dfs, SCENE_LABELS)
    if checks['warnings']:
        for warning in checks['warnings']:
            print(f"  {warning}")
    else:
        print("  ✓ Toutes les conditions de comparabilité sont remplies")
    print()
    
    # 3. Restructuration
    print("🔄 Étape 3: Restructuration des données...")
    df_wide = build_wide_table(dfs, SCENE_LABELS)
    df_wide.to_csv(os.path.join(OUTPUT_DIR, 'summary_wide.csv'), encoding='utf-8')
    print(f"  ✓ Table large générée: {OUTPUT_DIR}/summary_wide.csv")
    
    df_kpi = extract_kpi_table(df_wide, KPI_KEY_LIST)
    print(f"  ✓ {len(df_kpi)} KPI clés extraits")
    print()
    
    # 4. Métriques dérivées
    print("➕ Étape 4: Calcul des métriques dérivées...")
    df_kpi = compute_derived_metrics(df_kpi, df_wide)
    print(f"  ✓ Métriques dérivées ajoutées")
    
    # Scores
    df_scores = None
    if SCORE_WEIGHTS:
        df_scores = compute_scores(df_kpi, SCORE_WEIGHTS)
        if df_scores is not None:
            print(f"  ✓ Scores multi-objectif calculés")
    print()
    
    # Sauvegarder KPI
    df_kpi.to_csv(os.path.join(OUTPUT_DIR, 'kpi_key.csv'), index=False, encoding='utf-8')
    print(f"  ✓ KPI sauvegardés: {OUTPUT_DIR}/kpi_key.csv\n")
    
    # 5. Analyse des zones
    print("🗺️ Étape 5: Analyse des zones...")
    zone_data = extract_zone_data(df_wide, ZONES)
    zone_data['conversion'].to_csv(os.path.join(OUTPUT_DIR, 'zones_conversion.csv'), index=False, encoding='utf-8')
    zone_data['ventes'].to_csv(os.path.join(OUTPUT_DIR, 'zones_ventes.csv'), index=False, encoding='utf-8')
    print(f"  ✓ Données de zones extraites")
    print()
    
    # 6. Visualisations
    print("📊 Étape 6: Génération des visualisations...")
    
    # Graphiques KPI
    for kpi in KPI_KEY_LIST:
        cat, ind = kpi.split('::')
        safe_name = f"{cat}_{ind}".replace('::', '_').replace('/', '_')
        fig_path = os.path.join(FIGURES_DIR, f'kpi_{safe_name}.png')
        plot_kpi_comparison(df_kpi, kpi, fig_path)
    
    # Scatter plots
    scatter_pairs = [
        ('Parcours::IndiceDetour', 'Global::VolumeVentes'),
        ('Caisses::Saturations', 'Performance::DebitClients'),
        ('Caisses::AttenteMoyenneSecondes', 'Layout::IndiceFrustrationAttente')
    ]
    
    for x_kpi, y_kpi in scatter_pairs:
        x_safe = x_kpi.split('::')[1]
        y_safe = y_kpi.split('::')[1]
        fig_path = os.path.join(FIGURES_DIR, f'scatter_{x_safe}_vs_{y_safe}.png')
        plot_scatter(df_kpi, x_kpi, y_kpi, fig_path)
    
    # Zones
    plot_zone_heatmap(zone_data['conversion'], 
                     os.path.join(FIGURES_DIR, 'zone_conversion_heatmap.png'))
    plot_zone_sales(zone_data['ventes'], 
                   os.path.join(FIGURES_DIR, 'zone_ventes_comparison.png'))
    
    print()
    
    # 7. Rapport
    print("📝 Étape 7: Génération du rapport...")
    report_path = os.path.join(OUTPUT_DIR, 'report.md')
    generate_report(df_kpi, df_wide, checks, zone_data, df_scores, report_path)
    print()
    
    # 8. Interface HTML
    print("🌐 Étape 8: Génération de l'interface HTML...")
    html_path = os.path.join(OUTPUT_DIR, 'index.html')
    build_html(df_kpi, html_path, FIGURES_DIR)
    print()
    
    # Résumé
    print("=" * 70)
    print("✅ ANALYSE TERMINÉE AVEC SUCCÈS")
    print("=" * 70)
    print(f"\n📁 Tous les fichiers ont été générés dans: {OUTPUT_DIR}/")
    print(f"   - summary_wide.csv")
    print(f"   - kpi_key.csv")
    print(f"   - report.md")
    print(f"   - index.html")
    print(f"   - figures/*.png ({len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])} fichiers)\n")
    
    # Optionnel : démarrer le serveur web
    if AUTO_OPEN_BROWSER:
        try:
            # Ouvrir le fichier HTML directement
            html_full_path = os.path.abspath(html_path)
            webbrowser.open(f'file://{html_full_path}')
            print(f"🌐 Page HTML ouverte dans le navigateur")
            print(f"   Pour un serveur web interactif, modifiez AUTO_OPEN_BROWSER dans le script.\n")
        except Exception as e:
            print(f"⚠️ Impossible d'ouvrir le navigateur automatiquement: {e}\n")


if __name__ == '__main__':
    main()
