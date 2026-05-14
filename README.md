# SmartPuzzleAI - Projet Python local

Ce projet transforme le notebook `smartpuzzleai_final_merged.py` en application Python locale avec Gradio.

Le notebook original reste conserve tel quel. Le projet local reprend la partie utile de demo jusqu'au puzzle, dessin et comparaison, en retirant uniquement la Phase 5 de generation Stable Diffusion / LoRA.

## Fonctionnalites finales

- Chargement du dataset local depuis `Kaggle Children Drawings Dataset (`vishmiperera/children-drawings``
- Generation d'un puzzle depuis une image aleatoire
- Pieces de puzzle melangees
- Une piece manquante cachee dans le puzzle
- Affichage agrandi du puzzle
- Aucun affichage de l'image de reference
- Aucun affichage du nom de fichier ou de l'emotion
- Zone de dessin conservee
- Comparaison du dessin avec la vraie piece gardee en memoire
- Feedback de similarite comme dans la logique du notebook

## Structure

- `app.py` : interface Gradio finale.
- `smartpuzzleai/puzzle.py` : logique puzzle, piece cachee, dessin et comparaison.
- `smartpuzzleai/config.py` : chemins locaux du projet.
- `smartpuzzleai/siamese.py` : architecture Siamese conservee.
- `archive/` : dataset local.
- `SmartPuzzleAI_merged_model_final/` : modeles telecharges conserves dans le dossier.
- `outputs/` : fichiers generes pendant l'execution.

## Installation

```powershell
pip install -r requirements.txt
```

## Lancement

```powershell
python app.py
```

Gradio affichera une URL locale, generalement :

```text
http://127.0.0.1:7860
```

## Notes

La Phase 5 Stable Diffusion / LoRA a ete retiree de l'application locale pour eviter les problemes de chargement et de connexion. Les dossiers de modeles restent presents, mais l'app finale se concentre sur le puzzle, le dessin et la comparaison.
