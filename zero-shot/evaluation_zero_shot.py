import argparse
import os
import sys

import pandas as pd
import torch
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# Dataset pour charger les images de test avec leurs labels entiers
class ZeroShotDataset(Dataset):
    def __init__(self, path_dataset, data_root, classes, preprocess, niveau_exactitude="phylum"):
        df = pd.read_csv(path_dataset)

        # Si la colonne de label existe, l'utiliser directement
        # sinon extraire le premier mot de la caption comme classe (nom du genre)
        if niveau_exactitude in df.columns:
            df = df.dropna(subset=[niveau_exactitude])
            df['__label__'] = df[niveau_exactitude]
        else:
            df['__label__'] = df['caption'].str.split().str[0]

        # Ne garder que les lignes dont la classe est dans la liste des classes connues
        df = df[df['__label__'].isin(classes)].reset_index(drop=True)

        self.images = df['filepath'].tolist()
        self.labels = [classes.index(c) for c in df['__label__'].tolist()]
        self.data_root = data_root
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


# dans niveau d'exactitude soit je peux prendre le phylum ou bien class
"""
phylum a que 6 valeurs mais les autres on plus ce qui fait que les valeur d'exactitude vont etre de plus en plus petites
en utilisant les template suivant je peux remplacer les valeur entre parenthse avec les valeur des classe
"""

def generation_embedding_text_zero_shot(model, tokenizer, path_dataset, device, niveau_exactitude = "phylum"):
  templates_generale = []

  # importer le dataset et utilisé la classe que je veux
  df = pd.read_csv(path_dataset)

  # Si la colonne de label existe, l'utiliser directement
  # sinon extraire le premier mot de la caption comme classe (nom du genre)
  if niveau_exactitude in df.columns:
    df = df.dropna(subset=[niveau_exactitude])
    classes = sorted(df[niveau_exactitude].unique())
  else:
    df['__label__'] = df['caption'].str.split().str[0]
    classes = sorted(df['__label__'].unique())

  #print(len(classes))
  #print(classes)

  #-------------------------------------------
  # Templates
  with torch.no_grad():
    for ma_class in classes:
      templates = [
        # général underwater
        f"an underwater photo of a {ma_class}",
        f"a photo of a {ma_class} in the ocean",
        f"a marine organism: {ma_class}",

        # contexte scientifique
        f"a marine biology specimen of {ma_class}",
        f"a taxonomic image of {ma_class}",
        f"a zoological observation of {ma_class}",

        # environnement marin réaliste
        f"a {ma_class} swimming in deep sea water",
        f"a {ma_class} in a coral reef ecosystem",
        f"a {ma_class} on the ocean floor",

        # style caméra / exploration
        f"a deep sea exploration image of {ma_class}",
        f"an ROV captured image of {ma_class} underwater"
      ]
      # tokenization par batch de phrases
      templates_tokenizer = tokenizer(templates).to(device)

      # encoder le text
      templates_embedding = model.encode_text(templates_tokenizer)

      # L2 normalization recommender dans la destillation prcq cela nous evite qu'un prompt domine l'autre
      # et aussi rend le mean stable
      templates_embedding_normalize = templates_embedding / templates_embedding.norm(dim=-1, keepdim=True)

      # faire le mean pooling : faire la moyenne de tous les embedding de chaque phrase dans template
      template_encoded = templates_embedding_normalize.mean(dim=0)

      # L2 normalization
      vector_normalized = template_encoded / template_encoded.norm(dim=-1, keepdim=True)

      # stocker les resultats dans un dictionnaire avec les nom de classes
      templates_generale.append(vector_normalized)

    # transformet en une matrice
  text_features_final = torch.stack(templates_generale).to(device)

  return text_features_final, classes

def evaluate_zero_shot(model, preprocess, text_features, classes, dataloader, device, label_column="label"):

    model.eval()
    correct = 0
    total = 0
    # stocker tous les embeddings image pour pouvoir les comparer avec le student
    all_image_features = []

    with torch.no_grad():

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 2. Encode image batch
            image_features = model.encode_image(images)

            # 3. Normalize image features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 4. Similarity (batch × classes)
            logits = image_features @ text_features.T

            # 5. Prediction
            preds = torch.argmax(logits, dim=1)

            # 6. Accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # stocker les features normalisees pour la comparaison teacher/student
            all_image_features.append(image_features.cpu())

    accuracy = correct / total
    # retourner aussi les features pour la comparaison inter-modele
    all_image_features = torch.cat(all_image_features, dim=0)
    return accuracy, all_image_features


def comparer_student_teacher(model_student, preprocess_student, dataloader_student,
                              features_teacher, device):
    """
    Pour chaque batch on calcule la ressemblance (similarite cosinus)
    entre les embeddings image du student et ceux du teacher.
    Le teacher represente 100% — plus la similarite est proche de 1,
    plus le student a appris le meme espace de representation.
    """
    model_student.eval()
    similarites_par_batch = []
    idx_debut = 0

    with torch.no_grad():

        for images, _ in dataloader_student:
            images = images.to(device)
            batch_size = images.size(0)

            # encoder le batch avec le student
            features_student_batch = model_student.encode_image(images)
            features_student_batch = features_student_batch / features_student_batch.norm(dim=-1, keepdim=True)

            # recuperer les features teacher correspondantes (meme batch, meme ordre)
            features_teacher_batch = features_teacher[idx_debut : idx_debut + batch_size].to(device)
            idx_debut += batch_size

            # similarite cosinus par paire student[i] . teacher[i]
            # les deux etant deja normalises, le produit scalaire = cosinus
            similarite_batch = (features_student_batch * features_teacher_batch).sum(dim=-1)

            # moyenne sur le batch
            similarites_par_batch.append(similarite_batch.mean().item())

    # moyenne globale sur tous les batchs
    similarite_moyenne = sum(similarites_par_batch) / len(similarites_par_batch)
    return similarite_moyenne, similarites_par_batch


def parse_args(args):
    parser = argparse.ArgumentParser(description="Évaluation zero-shot classification organismes marins")
    # ---- student (modele a evaluer) ----
    parser.add_argument("--model",              type=str, required=True,  help="Nom de l'architecture student (ex: ViT-T-16)")
    parser.add_argument("--model-checkpoint",   type=str, required=True,  help="Chemin vers le checkpoint du student")
    # ---- teacher (reference = 100%) ----
    parser.add_argument("--t-model",            type=str, default=None,   help="Nom de l'architecture teacher (ex: ViT-B-16)")
    parser.add_argument("--t-model-checkpoint", type=str, default=None,   help="Chemin vers le checkpoint du teacher")
    # ---- donnees ----
    parser.add_argument("--dataset",            type=str, required=True,  help="Chemin vers le CSV de test")
    parser.add_argument("--data-root",          type=str, default="",     help="Repertoire racine des images")
    parser.add_argument("--label-column",       type=str, default="phylum",
                        help="Colonne label dans le CSV. Si absente, utilise le 1er mot de la caption comme classe")
    parser.add_argument("--batch-size",         type=int, default=32,     help="Taille des batches")
    parser.add_argument("--workers",            type=int, default=4,      help="Nombre de workers DataLoader")
    parser.add_argument("--seed",               type=int, default=42,     help="Graine aleatoire")
    return parser.parse_args(args)


def charger_model(nom_model, path_model, device):
    """Charge un modele open_clip depuis un checkpoint en gerant tous les formats du pipeline."""
    # charger le model
    model, _, preprocess = open_clip.create_model_and_transforms(nom_model)
    state_dict = torch.load(path_model, map_location = device)

    # gérer les différents formats de checkpoint sauvegardés par le pipeline d'entraînement
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    # utilisé les poid deja appris
    model.load_state_dict(state_dict)

    # mettre le model en evaluation
    model.eval()
    model.to(device)
    tokenizer = open_clip.get_tokenizer(nom_model)

    return model, preprocess, tokenizer


def main(args):
    args = parse_args(args)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NOM_MODEL    = args.model
    PATH_MODEL   = args.model_checkpoint
    PATH_DATASET = args.dataset
    niveau_exactitude = args.label_column

    # ------------------------------------------------------------------ #
    # Evaluation du teacher (reference = 100%) si fourni                  #
    # ------------------------------------------------------------------ #
    accuracy_teacher = None
    if args.t_model is not None and args.t_model_checkpoint is not None:

        NOM_MODEL_TEACHER  = args.t_model
        PATH_MODEL_TEACHER = args.t_model_checkpoint

        print(f"\n--- Teacher : {NOM_MODEL_TEACHER} ---")
        t_model, t_preprocess, t_tokenizer = charger_model(NOM_MODEL_TEACHER, PATH_MODEL_TEACHER, device)

        print(f"Génération des embeddings texte zero-shot (niveau : {niveau_exactitude})...")
        text_features_final, classes = generation_embedding_text_zero_shot(
            t_model, t_tokenizer, PATH_DATASET, device, niveau_exactitude
        )
        print(f"  → {len(classes)} classes trouvées : {classes[:5]}{'...' if len(classes) > 5 else ''}")

        # Création du DataLoader pour les images de test avec labels entiers
        dataset_test = ZeroShotDataset(PATH_DATASET, args.data_root, classes, t_preprocess, niveau_exactitude)
        dataloader   = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        print(f"Évaluation zero-shot teacher sur {len(dataset_test)} images...")
        accuracy_teacher, features_teacher = evaluate_zero_shot(t_model, t_preprocess, text_features_final, classes, dataloader, device, niveau_exactitude)

        # liberer la memoire GPU avant de charger le student (features_teacher restent en CPU)
        del t_model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Evaluation du student                                               #
    # ------------------------------------------------------------------ #
    print(f"\n--- Student : {NOM_MODEL} ---")
    model, preprocess, tokenizer = charger_model(NOM_MODEL, PATH_MODEL, device)

    print(f"Génération des embeddings texte zero-shot (niveau : {niveau_exactitude})...")
    text_features_final, classes = generation_embedding_text_zero_shot(
        model, tokenizer, PATH_DATASET, device, niveau_exactitude
    )
    print(f"  → {len(classes)} classes trouvées : {classes[:5]}{'...' if len(classes) > 5 else ''}")

    # Création du DataLoader pour les images de test avec labels entiers
    dataset_test = ZeroShotDataset(PATH_DATASET, args.data_root, classes, preprocess, niveau_exactitude)
    dataloader   = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print(f"Évaluation zero-shot student sur {len(dataset_test)} images...")
    accuracy_student, features_student = evaluate_zero_shot(model, preprocess, text_features_final, classes, dataloader, device, niveau_exactitude)

    # ------------------------------------------------------------------ #
    # Affichage de la comparaison teacher vs student                      #
    # le teacher represente 100% : on mesure combien le student a retenu  #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*50}")
    print(f"  Résultats Zero-Shot — comparaison KD")
    print(f"{'='*50}")
    print(f"  Dataset     : {PATH_DATASET}")
    print(f"  Niveau      : {niveau_exactitude}")
    print(f"  Nb classes  : {len(classes)}")
    print(f"{'='*50}")

    if accuracy_teacher is not None:
        # retention = ce que le student a retenu par rapport au teacher (teacher = 100%)
        retention = (accuracy_student / accuracy_teacher) * 100 if accuracy_teacher > 0 else 0.0

        print(f"  Teacher ({args.t_model:<12}) : {accuracy_teacher * 100:6.2f}%  (référence = 100%)")
        print(f"  Student ({NOM_MODEL:<12}) : {accuracy_student * 100:6.2f}%  ({retention:.1f}% du teacher)")
        print(f"{'='*50}")
        print(f"  Ecart absolu              : {(accuracy_teacher - accuracy_student) * 100:+.2f}%")
        print(f"  Retention student/teacher : {retention:.1f}%")

        # ---- similarite cosinus student/teacher par batch ----
        # pour chaque batch on calcule la ressemblance entre les representations
        # du student et du teacher sur les memes images
        print(f"\n  Calcul de la ressemblance student/teacher par batch...")
        similarite_moyenne, similarites_par_batch = comparer_student_teacher(
            model, preprocess, dataloader, features_teacher, device
        )
        print(f"  Ressemblance cosinus student/teacher :")
        print(f"    Moyenne globale  : {similarite_moyenne:.4f}  ({similarite_moyenne * 100:.2f}% d'alignement)")
        print(f"    Min par batch    : {min(similarites_par_batch):.4f}")
        print(f"    Max par batch    : {max(similarites_par_batch):.4f}")

    else:
        print(f"  Student ({NOM_MODEL:<12}) : {accuracy_student * 100:6.2f}%")

    print(f"{'='*50}\n")

    return accuracy_student, accuracy_teacher


if __name__ == "__main__":
    main(sys.argv[1:])
