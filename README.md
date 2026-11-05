# CTTH — Altitude du sommet des nuages par apprentissage automatique (MTG-FCI × EarthCARE)

Projet M2 Ingénierie Mathématique et Science des Données, UHA (IRIMAS, équipe IMTIS), en collaboration avec Météo-France (CNRM).
Encadrants : Jean-Baptiste Courbot, Ali Moukadem (UHA) ; Tony Le Bastard, Emmanuel Fontaine (Météo-France).
Étudiants : Kossi Noumagno, Rayhan Saouir.

Ce dossier rassemble le travail réalisé par Kossi entre le 29 octobre et le 2 novembre 2026, pour que Rayhan puisse le reprendre et le poursuivre.

---

## 1. Le projet en bref

**Objectif** : estimer l'altitude du sommet des nuages à partir des images infrarouges du satellite géostationnaire **MTG-FCI** (une image toutes les 10 minutes), en apprenant sur les mesures du radar **EarthCARE-CPR** (vérité terrain précise, mais seulement sous sa trace). La référence opérationnelle à battre est le produit **CTTH du SAFNWC**.

**Résultat principal** (validation croisée sur les 13 jours, scénario réaliste, sans fuite d'information) :

| Modèle | MAE | Commentaire |
|---|---|---|
| SAFNWC (outil opérationnel) | 1 220 m | Référence, hors nuages fractionnés |
| Gradient boosting, meilleures variables (famille F) | 727 m | Outil de tri des variables |
| **MLP corrigé (PyTorch), famille C** | **647 m** | **Meilleur modèle actuel, −47 % par rapport au SAFNWC** |
| MLP corrigé avec la vignette corrigée de Tony (fuite) | 591 m | Plafond « alignement parfait », pas utilisable en opérationnel |

---

## 2. Ce qui a été fait, dans l'ordre

| Étape | Pourquoi | Principal résultat | Fichier |
|---|---|---|---|
| 1. Revue du code de Tony (`ctth_IA.py`) | Savoir si le score de référence est fiable | 22 problèmes, dont 4 critiques : « RMSE » et « R2 » faux, activation manquante, modèle rechargé en silence | `docs/01_Revue_ctth_IA_limites_et_bugs.pdf` |
| 2. EDA du jeu V1 | Comprendre les données de départ | Données propres ; pixels voisins très corrélés (0,978) : découpage obligatoire par jour | `01_EDA_V1.ipynb` |
| 3. Analyse du jeu V2 et de la parallaxe | Les vignettes V1 ont été recalées **avec la cible** (fuite) | Convention des axes trouvée (axe 0 = longitude) ; vignette V1 = transposée de la sous-fenêtre corrigée (99,97 %) | `docs/01_Revue…pdf`, section P1 |
| 4. Nommage explicite des 45 variables | Rendre visibles les variables interdites (dérivées de la cible) | Dictionnaire complet ; garde-fou automatique dans le code | `docs/02_Dictionnaire_variables_V2.pdf` |
| 5. Module commun `ctth_data.py` | Ne plus recopier le code d'un notebook à l'autre | Chargement, variables, vraies métriques, validation croisée par jour | `ctth_data.py` |
| 6. EDA du jeu V2 | Préparer la modélisation et les méthodes spatiales | Fuite = 46 m ; voisinage brut 5×5 = −234 m ; champ dominé par une pente à grande échelle | `02_EDA_V2.ipynb` |
| 7. Descripteurs spatiaux (Gabor, scattering) | Piste proposée par les encadrants | Gabor −10 m (non significatif) ; scattering inutile ; la texture des canaux vapeur d'eau est la variable la plus liée à l'altitude | `03_Descripteurs_spatiaux.ipynb` |
| 8. MLP PyTorch | Mesurer le vrai score du MLP de Tony et le comparer à l'identique | Réplique fidèle (574 m contre 564 m) ; MLP ≫ gradient boosting (−132 m) ; MLP corrigé répare les nuages très hauts (1 070 → 549 m) | `04_MLP_PyTorch.ipynb`, `ctth_models.py` |
| 9. Rapport et présentation | Communiquer aux encadrants et à un public non technique | Rapport d'avancement ; présentation de 8 diapositives avec notes | `docs/03_…pdf`, `docs/04_…pptx` |

---

## 3. Organisation du dossier

```
CTTH_projet/
├── README.md                       ← ce fichier
├── requirements.txt                ← dépendances Python
├── ctth_data.py                    ← module commun : données, variables, métriques, validation croisée
├── ctth_models.py                  ← module MLP PyTorch (réplique de Tony, MLP corrigé)
├── 01_EDA_V1.ipynb                 ← analyse exploratoire du jeu V1 (archive)
├── 02_EDA_V2.ipynb                 ← analyse exploratoire du jeu V2 + familles de variables A à E
├── 03_Descripteurs_spatiaux.ipynb  ← Gabor et scattering, familles E à H
├── 04_MLP_PyTorch.ipynb            ← MLP de Tony reproduit, MLP corrigé, deux scénarios
├── Data/
│   ├── LISEZ-MOI.md                ← où trouver les fichiers .pkl
│   └── spatial_features_v2.parquet ← cache des descripteurs spatiaux (évite 16 min de calcul)
├── Runs/                           ← résultats de validation croisée enregistrés (évite ≈ 80 min de calcul)
├── scripts/                        ← scripts d'expérience lancés en dehors des notebooks
├── docs/                           ← revue du code, dictionnaire, rapport, présentation
└── reference/ctth_IA_original_Tony.py  ← script d'origine, pour mémoire
```

Tous les notebooks sont **déjà exécutés** : on peut lire les résultats sans rien lancer.

---

## 4. Démarrage

1. **Installer les dépendances** : `pip install -r requirements.txt`
2. **Ajouter les données** dans `Data/` (voir `Data/LISEZ-MOI.md`) :
   - `data_cloud_top.pkl` (V1, 125 Mo), seulement pour `01_EDA_V1.ipynb` et `scripts/check_tony_v1.py` / `tune_v1.py` ;
   - `data_cloud_top_V2.pkl` (V2, 626 Mo), pour tout le reste.
3. **Ouvrir Jupyter depuis ce dossier** (les chemins sont relatifs : `./Data/…`, `./Runs/…`), puis exécuter les notebooks dans l'ordre.

Temps d'exécution sur un ordinateur portable (2 cœurs CPU), grâce aux caches fournis :

| Notebook | Avec les caches | Sans cache |
|---|---|---|
| 01_EDA_V1 | < 1 min | identique |
| 02_EDA_V2 | ≈ 10 min (section 7) | identique (`RUN_CV = False` pour la sauter) |
| 03_Descripteurs_spatiaux | ≈ 12 min | + 16 min pour recalculer Gabor et scattering |
| 04_MLP_PyTorch | < 1 min | ≈ 80 min (`RUN_TRAINING = True`, résultats dans `Runs/`) |

Pour refaire un entraînement, supprimer le fichier `.npz` correspondant dans `Runs/`.

---

## 5. Utiliser le module commun

```python
import ctth_data as cd
df, win = cd.load_v2('./Data/data_cloud_top_V2.pkl')   # df : variables renommées ; win : (N, 6, 11, 11)
df = cd.build_features(df, win)                          # toutes les variables de l'EDA (≈ 180 colonnes)

feats = cd.FAMILIES_V2['C']                              # familles A à H, voir section 6
cd.check_features(feats)                                 # refuse toute variable dérivée de la cible

import lightgbm as lgb
res = cd.cross_validate_by_day(df, feats, lambda: lgb.LGBMRegressor(objective='l1'))
print(res['summary'])                                    # MAE, biais, vrai RMSE, vrai R², écart-type entre jours

import ctth_models as cm                                 # MLP PyTorch
run = cm.cross_validate_mlp(df, feats, cm.CORRECTED_CONFIG)
print(cd.paired_comparison(res['by_day'], run['by_day']))   # comparaison jour par jour + test du signe
```

Fonctions principales :

| Fonction | Rôle |
|---|---|
| `load_v2` | Charge la V2, renomme les colonnes, convertit 63 535 (SAFNWC absent) en NaN, construit le cube des fenêtres |
| `build_features` | Calcule les variables : stats 5×5 et 11×11 brutes, contrastes physiques, gradient, ondelettes, inversion ERA5 |
| `spatial_features` | Descripteurs Gabor et scattering (avec cache) |
| `patch_raw`, `patch_corrected`, `detrend_plane` | Vignettes brutes ou corrigées ; retrait de la pente avant analyse de texture |
| `check_features` | Garde-fou : refuse `p5corr`, `parallax_shift`, `earthcare` en scénario opérationnel |
| `metrics`, `metrics_by` | Vraies métriques, globales ou par type de nuage / tranche d'altitude |
| `cross_validate_by_day` | Validation croisée « un jour = un pli » (13 plis) |
| `paired_comparison` | Gain jour par jour entre deux modèles, avec test du signe |
| `plot_hist2d` | Histogramme 2D prédiction / EarthCARE |
| `cm.cross_validate_mlp` | Idem pour un MLP, avec un jour de validation interne pour l'arrêt précoce |

---

## 6. Conventions et règles importantes

**Deux scénarios, à ne jamais mélanger**

- **Conditions de Tony** : vignette 5×5 **corrigée de la parallaxe à l'aide de l'altitude EarthCARE**. C'est une fuite d'information : utile seulement pour se comparer à Tony.
- **Opérationnel** : fenêtre 11×11 **brute**. C'est le seul scénario comparable au SAFNWC et utilisable en conditions réelles.

**Variables interdites en opérationnel** : toute colonne dont le nom contient `p5corr`, `parallax_shift` ou `earthcare`. Le garde-fou `check_features` les bloque.

**Familles de variables** (emboîtées) :

| Famille | Contenu | Nb |
|---|---|---|
| A | Pixel au sol + ciel clair + ERA5 | 19 |
| B | A + stats 5×5 **corrigée** (fuite) | 37 |
| C | A + stats 5×5 brute | 37 |
| D | C + stats 11×11 brute | 61 |
| E | D + physique (contrastes, split-window, inversion ERA5) + texture simple (gradient, ondelettes) | 79 |
| F | E + Gabor | 115 |
| G | E + scattering | 115 |
| H | E + Gabor + scattering | 151 |

`cd.REFERENCE_FEATURES` = les 37 variables exactes du script de Tony (avec fuite).

**Fenêtre 11×11** : axe 0 = longitude, axe 1 = latitude, centre `[5, 5]` = pixel au sol. Pixel corrigé : `[5 + parallax_shift_lon_px, 5 + parallax_shift_lat_px]`. Le sens du nord n'est pas encore confirmé.

**Évaluation** : toujours en validation croisée par jour, et comparer deux modèles **jour par jour** (`paired_comparison`). L'écart-type de la MAE entre jours est d'environ 80 à 90 m : un écart de quelques dizaines de mètres sur un seul jour ne prouve rien. Les jours voisins se ressemblent ; pour un chiffre final, prévoir aussi une validation avec un tampon de ±1 jour (`scripts/temporal_cv.py` : environ +24 m).

---

## 7. Résultats détaillés

**Gradient boosting par famille** (`02_EDA_V2`, `03_Descripteurs_spatiaux`) :

| A | B (fuite) | C | D | E | F | G | H | SAFNWC |
|---|---|---|---|---|---|---|---|---|
| 1 007 m | 728 m | 772 m | 762 m | 736 m | 727 m | 730 m | 729 m | 1 220 m |

**MLP** (`04_MLP_PyTorch`) :

| Scénario | Gradient boosting | MLP Tony (réplique) | MLP corrigé |
|---|---|---|---|
| Conditions de Tony (fuite, 37 var.) | 720 m | 584 m | 591 m |
| Opérationnel, famille C (37 var.) | 772 m | 647 m | **647 m** |
| Opérationnel, famille F (115 var.) | 727 m | 646 m | 648 m |

Points clés :
- Le MLP bat le gradient boosting de 75 à 132 m, 13 jours sur 13 : le modèle final doit être un réseau de neurones.
- Le MLP corrigé (même architecture, cible standardisée) n'améliore pas la moyenne mais corrige les nuages très hauts (type 9 : 1 070 → 549 m, meilleur que le SAFNWC à 707 m).
- La fuite coûte 56 à 65 m au MLP. **C'est l'écart que le CNN doit chercher à combler** en retrouvant seul la position du nuage dans la fenêtre.
- Les descripteurs Gabor n'aident pas le MLP.

---

## 8. Scripts (`scripts/`)

À lancer depuis le dossier du projet (`python scripts/<nom>.py`) :

| Script | Rôle | Durée (2 cœurs) |
|---|---|---|
| `check_tony_v1.py` | Vérifie que la réplique PyTorch reproduit le MLP de Tony sur V1 (≈ 574 m) | ≈ 5 min |
| `tune_v1.py` | Teste les corrections du MLP une par une, choix sur le jour de validation | ≈ 15 min |
| `temporal_cv.py` | Compare la validation croisée normale, avec tampon, « passé seulement » et par bloc | ≈ 10 min |
| `run_mlp_cv.py` | Lance toutes les validations croisées des MLP et enregistre les résultats dans `Runs/` | ≈ 50 min |
| `ablation_familles_mlp.py` | Retire une famille de variables à la fois du MLP et mesure la hausse d'erreur (`--rapide` pour un test sur 3 jours) | ≈ 30 min (rapide : ≈ 7 min) |

---

## 9. Questions ouvertes (à poser à Tony Le Bastard)

- Sens du nord et de l'est dans la fenêtre 11×11 (important pour Gabor et l'augmentation de données des CNN).
- Origine des quelque 5 100 lignes hors type 10 ajoutées dans la V2.
- Possibilité d'obtenir latitude, longitude et angle de visée de chaque pixel (correction de parallaxe en deux passes, analyse par région).
- Le type de nuage SAFNWC peut-il servir d'entrée, ou seulement de référence ?

## 10. Prochaine étape : les CNN

- Petit CNN sur le cube `(6, 11, 11)` + branche dense pour ERA5 et ciel clair.
- Même validation croisée par jour ; base de comparaison : **MLP corrigé, famille C, 647 m**.
- Objectif : récupérer les ≈ 60 m perdus sans correction de parallaxe.
- Précautions : réseau compact (peu de données indépendantes), augmentation par rotation à tester prudemment (le décalage de parallaxe a une direction), GPU conseillé (Kaggle ou Colab).
