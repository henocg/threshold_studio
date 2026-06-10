# Threshold Studio — LSN Ré Walbaum

Tableau de bord interactif Python/Streamlit pour la **détection du seuil de sinistre grave** (approche POT) et le calcul des niveaux de retour en réassurance non-proportionnelle.

---

## Démarrage rapide (1 commande)

```bash
# 1. Installer les dépendances (une seule fois)
pip install -r requirements.txt

# 2. Lancer le tableau de bord
streamlit run app.py
```

Le navigateur s'ouvre automatiquement sur `http://localhost:8501`.

---

## Utilisation

1. **Charger vos données** — glissez un fichier CSV ou Excel dans la zone de dépôt (colonne de charges en €)
2. **Sélectionner la colonne** — choisir la colonne de charges brutes ou indexées
3. **Parcourir les 6 méthodes** — onglets MRL, Hill, DEH, Gerstengarbe, GPD Stabilité, MCDA
4. **Ajuster le seuil u*** — slider manuel ou cliquer sur un seuil automatique
5. **Lire les diagnostics** — KS, Anderson-Darling, AIC, niveaux de retour T = 10 … 500 ans
6. **Exporter** — bouton "Exporter résultats (CSV)"

---

## Structure des fichiers

```
threshold_studio/
├── app.py                  ← Application principale (point d'entrée)
├── requirements.txt        ← Dépendances Python
├── .streamlit/
│   └── config.toml         ← Thème couleurs LSN Ré Walbaum
├── methods/                ← Un fichier par méthode EVT
│   ├── mrl.py              ← Mean Residual Life
│   ├── hill.py             ← Estimateur de Hill
│   ├── deh.py              ← Dekkers-Einmahl-de Haan
│   ├── gerstengarbe.py     ← Gerstengarbe Plot
│   ├── gpd_stability.py    ← Stabilité paramètres GPD
│   └── mcda.py             ← MCDA entropie Shannon
└── utils/
    └── gpd_utils.py        ← Ajustement GPD, niveaux de retour, tests
```

---

## Format du fichier de données attendu

| colonne          | type     | exemple        |
|------------------|----------|----------------|
| `charge_brute`   | float    | 125000         |
| `charge_indexée` | float    | 138500         |
| `date`           | date     | 2018-03-14     |
| `branche`        | texte    | Incendie       |

Seule la colonne numérique de charges est obligatoire. Les autres sont ignorées.

---

## Dépendances

```
streamlit    ≥ 1.35
pandas       ≥ 2.0
numpy        ≥ 1.26
scipy        ≥ 1.12
plotly       ≥ 5.20
openpyxl     ≥ 3.1   (lecture Excel .xlsx)
xlrd         ≥ 2.0   (lecture Excel .xls)
```

---

## Auteur

Hénoc Gakpeto — Stage actuariat, LSN Ré Walbaum – DiotSiaci Group (2026)
