# Détection d'émotions dans les tweets : TF-IDF vs BERT

Projet NLP réalisé dans le cadre du cours **NLP — AI 2026–2027** à l'**ESSAI** (École Supérieure de la Statistique et de l'Analyse de l'Information, Tunis).

**Auteurs** : Asma Allaigui · Nourchene Boumaiza

---

## Problématique

Détecter une émotion (joie, tristesse, colère, surprise…) dans un tweet est plus difficile qu'une simple analyse de sentiment positif / négatif : les textes sont **très courts**, écrits dans un **langage bruité** (argot, émoticônes, élongations comme *soooo*) et annotés de façon **subjective**.

> **Question** : un Transformer pré-entraîné comme **BERT**, qui comprend le contexte, fait-il nettement mieux que des baselines classiques **TF-IDF + modèle linéaire** sur ce type de données ?

---

## Données

- **Dataset** : CrowdFlower *« Emotion in Text »*, soit 40 000 tweets en anglais (2009) annotés en 13 émotions — [Kaggle](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)
- **Constats de l'exploration** :
  - **fort déséquilibre** des classes (ratio ≈ 78:1 entre `neutral` et `anger`) ;
  - **bruit d'annotation** : des tweets identiques portent des labels différents.

---

## Méthodologie

### 1. Préparation des données
- **Suppression des doublons** (pour éviter toute fuite entre train et test) et des tweets aux labels contradictoires.
- **Regroupement des 13 émotions en 6 macro-émotions** selon le modèle valence / activation de Russell : `sadness`, `neutral`, `joy`, `love`, `surprise`, `anger`. Le déséquilibre passe ainsi de 78:1 à ≈ 10:1.

### 2. Prétraitement adapté à Twitter (baselines TF-IDF)
- Décodage des entités HTML (`&lt;3` → `<3`)
- Émoticônes, « ! » et « ? » conservés sous forme de jetons (`xxsmile`, `xxheart`, `xxexcl`…)
- URLs et mentions → `xxurl`, `xxuser`
- Élongations réduites, contractions et argot normalisés (`can't` → `cannot`, `u` → `you`)
- Suppression des stopwords en **conservant les négations et les intensifieurs** (*not*, *so*, *very*)
- Lemmatisation avec spaCy

Pour **BERT**, le texte est conservé presque brut : le modèle exploite le contexte complet.

### 3. Modèles
| Modèle | Détails |
|---|---|
| **TF-IDF + Régression logistique** | uni- et bigrammes, *Grid Search* sur `C` et `class_weight`, validation croisée stratifiée à 5 plis |
| **TF-IDF + SVM linéaire** | même protocole |
| **BERT-base uncased** (fine-tuné) | 3 epochs, lr = 2e-5, loss pondérée par classe, meilleur epoch choisi sur un jeu de validation |

Le **déséquilibre** est traité par **pondération des classes** dans les trois modèles, plutôt que par oversampling ou undersampling.

### 4. Évaluation
- Métrique principale : **F1-macro**, où chaque émotion compte autant, quelle que soit sa taille.
- Split stratifié 80 / 20 ; le jeu de test n'est utilisé qu'une seule fois, pour la comparaison finale.

---

## Résultats

### F1 par émotion (jeu de test)

| Émotion | TF-IDF + LogReg | TF-IDF + SVM | BERT |
|---|---|---|---|
| sadness | 0.530 | **0.584** | 0.573 |
| neutral | 0.455 | 0.437 | **0.480** |
| joy | 0.436 | 0.467 | **0.473** |
| love | 0.431 | 0.431 | **0.450** |
| surprise | 0.165 | 0.152 | **0.208** |
| anger | 0.288 | 0.286 | **0.317** |
| **F1-macro** | 0.384 | 0.393 | **0.417** |

### Enseignements
- **BERT obtient le meilleur F1-macro (+2,4 points)** par rapport à la meilleure baseline, avec un gain surtout marqué sur les classes rares (`surprise`, `anger`).
- **La pondération des classes** est le réglage décisif pour les baselines (≈ +4 points de F1-macro).
- **Le prétraitement a un effet limité** sur la baseline (0,363 contre 0,357 en validation croisée, écart non significatif).
- **Principales confusions** : `anger` ↔ `sadness` et `joy` ↔ `love`, des émotions proches en valence.
- **La qualité des annotations est la principale limite** : beaucoup d'erreurs portent sur des tweets au label discutable.

---

## Reproduire le projet

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/Asma2201/emotion-detection-tweets.git
   cd emotion-detection-tweets
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. Télécharger le dataset (voir [`data/README.md`](data/README.md)).
4. Exécuter `emotion_detection.ipynb`. **Un GPU est recommandé** pour le fine-tuning de BERT (≈ 10 min sur un GPU Kaggle T4).

---

## Limites et perspectives

- Labels bruités et mono-label ; regroupement des classes discutable (`worry` est plutôt de la peur que de la tristesse).
- Un seul entraînement de BERT (sensible à la graine aléatoire).
- **Pistes** : modèles pré-entraînés sur des tweets (Twitter-RoBERTa, BERTweet), augmentation de données pour les classes rares, classification multi-label.

---

## Stack

Python · pandas · scikit-learn · spaCy · PyTorch · Hugging Face Transformers

## Références

- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
- Russell, J. A. (1980). *A circumplex model of affect.* Journal of Personality and Social Psychology.
- Barbieri et al. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.* Findings of EMNLP.
- Mohammad et al. (2018). *SemEval-2018 Task 1: Affect in Tweets.* SemEval.