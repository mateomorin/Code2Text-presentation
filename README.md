# Code2Text-presentation
Slides quarto pour présenter Code2Text.

Lien vers le site généré : https://mateomorin.github.io/Code2Text-presentation/.

# Structure
Le projet est constitué comme suit :
```
├── _extensions                         # Style des slides
├── notebooks                           # Code pour générer les graphiques (ATTENTION : besoin de la donnée de génération sur S3)
├── slides                              # Slides de présentation
└── sources
    ├── benchmarks                      # CSV de résultats ML
    ├── excalidraw_plots                # Schémas explicatifs
    ├── figures                         # Plots générés par le code
    ├── generation_details              # Données utiles pour avoir un aperçu de la génération/NAF 
    ├── images                          # Images utilisées pour le rapport
    └── overview_plots                  # Premiers plots pour étudier les générations au tout départ de Code2Text
```

# Comment faire tourner le code ?

Le code n'est pas fait pour être tourné depuis l'extérieur, il est simplement présent pour rendre compte de comment créer les figures, et pour qu'il soit facilement réexécutable par quiconque ayant accès au SSP Cloud avec les bons crédits.

Si c'est le cas, un fichier `requirements.txt` est présent au cas où pour installer les dépendances minimales, tous les codes sont ensuite indépendants.