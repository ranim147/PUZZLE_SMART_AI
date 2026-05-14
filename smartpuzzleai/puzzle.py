import base64
import html
import io
import random

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import DATASET_DIR, OUTPUTS_DIR

saved_puzzle_img = None
saved_missing_piece = None
saved_puzzle_info = None


def get_random_image_path():
    """Retourne le chemin d'une image aleatoire du dataset."""
    all_images = []
    for emotion in DATASET_DIR.iterdir():
        if emotion.is_dir():
            all_images += [
                item
                for item in emotion.iterdir()
                if item.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
            ]
    if not all_images:
        raise ValueError(f"Aucune image trouvee dans {DATASET_DIR}")
    return random.choice(all_images)


def create_puzzle_with_missing_piece(grid_size=(3, 3)):
    """
    Decoupe une image aleatoire en grille et masque une piece.
    Retourne : (puzzle_img, missing_piece, info_dict, original_img)
    """
    global saved_puzzle_img, saved_missing_piece, saved_puzzle_info

    image_path = get_random_image_path()
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Image illisible : {image_path}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    rows, cols = grid_size
    ph, pw = h // rows, w // cols

    mr = random.randint(0, rows - 1)
    mc = random.randint(0, cols - 1)
    y1, y2 = mr * ph, (mr + 1) * ph
    x1, x2 = mc * pw, (mc + 1) * pw

    puzzle_img = img.copy()
    puzzle_img[y1:y2, x1:x2] = [235, 235, 240]
    missing_piece = img[y1:y2, x1:x2].copy()

    saved_puzzle_img = puzzle_img.copy()
    saved_missing_piece = missing_piece.copy()
    saved_puzzle_info = {
        "image_name": image_path.name,
        "emotion": image_path.parent.name,
        "row": mr,
        "col": mc,
        "y1": y1,
        "y2": y2,
        "x1": x1,
        "x2": x2,
    }

    cv2.imwrite(
        str(OUTPUTS_DIR / "current_puzzle.jpg"),
        cv2.cvtColor(puzzle_img, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(OUTPUTS_DIR / "missing_piece_reference.jpg"),
        cv2.cvtColor(missing_piece, cv2.COLOR_RGB2BGR),
    )

    return puzzle_img, missing_piece, saved_puzzle_info, img


def feedback_engine(similarity_score: float) -> dict:
    """
    Regles de feedback basees sur le score cosinus du Siamese Network.
    Retourne : niveau, message, conseil, emoji, couleur
    """
    sim_pct = max(0, int(similarity_score * 100))

    if similarity_score > 0.85:
        return {
            "niveau": "Excellent",
            "emoji": "*",
            "couleur": "#2ecc71",
            "score_pct": sim_pct,
            "message": f"Felicitations ! Ton dessin est quasi-identique ({sim_pct}%) !",
            "conseil": "Tu es un vrai artiste ! Essaie la piece suivante.",
        }

    if similarity_score > 0.65:
        return {
            "niveau": "Tres bien",
            "emoji": "+",
            "couleur": "#3498db",
            "score_pct": sim_pct,
            "message": f"Excellent travail ! Ton dessin ressemble beaucoup ({sim_pct}%)",
            "conseil": "Affine les contours pour etre encore plus precis !",
        }

    if similarity_score > 0.45:
        return {
            "niveau": "Bien",
            "emoji": ":)",
            "couleur": "#f39c12",
            "score_pct": sim_pct,
            "message": f"Pas mal ! Tu es sur la bonne voie ({sim_pct}%)",
            "conseil": "Regarde bien la couleur et la forme globale.",
        }

    return {
        "niveau": "Continue",
        "emoji": "art",
        "couleur": "#e74c3c",
        "score_pct": sim_pct,
        "message": f"Continue d'essayer ! Observe bien la reference ({sim_pct}%)",
        "conseil": "Commence par les grandes formes avant les details.",
    }


def generate_new_puzzle():
    """
    Genere un nouveau puzzle complet :
    - image aleatoire depuis dataset
    - decoupage puzzle
    - suppression d'une piece
    """
    puzzle_img, missing_piece, info, _original = create_puzzle_with_missing_piece()

    text_info = f"""
Puzzle genere !

Emotion detectee : {info['emotion']}
Dessinez la piece manquante dans la zone de dessin.
"""

    return Image.fromarray(puzzle_img), Image.fromarray(missing_piece), text_info


def image_editor_to_pil(image):
    if isinstance(image, dict):
        image = image.get("composite", None)
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype("uint8")).convert("RGB")
    return image.convert("RGB")


def compare_with_missing_piece(user_drawing):
    """
    Compare le dessin utilisateur avec la vraie piece manquante
    et retourne si cela correspond ou non.
    """
    global saved_missing_piece

    if user_drawing is None:
        return "Veuillez dessiner la piece manquante."

    if saved_missing_piece is None:
        return "Aucune piece de reference trouvee."

    try:
        user_img = image_editor_to_pil(user_drawing)
        if user_img is None:
            return "Aucun dessin detecte."

        if isinstance(saved_missing_piece, np.ndarray):
            target_img = Image.fromarray(saved_missing_piece.astype("uint8")).convert("RGB")
        else:
            target_img = saved_missing_piece.convert("RGB")

        user_img = user_img.resize((128, 128))
        target_img = target_img.resize((128, 128))

        user_array = np.array(user_img).astype(np.float32)
        target_array = np.array(target_img).astype(np.float32)

        diff = np.mean(np.abs(user_array - target_array))
        score = max(0, 100 - (diff / 2))

        if score >= 80:
            verdict = "Excellent ! Votre dessin correspond tres bien."
        elif score >= 60:
            verdict = "Bon travail ! Le dessin est assez proche."
        elif score >= 40:
            verdict = "Partiellement correct, peut etre ameliore."
        else:
            verdict = "Le dessin ne correspond pas assez a la piece reelle."

        return f"""
Score de similarite : {score:.2f}%

{verdict}
"""

    except Exception as exc:
        return f"Erreur : {str(exc)}"


def pil_to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def create_shuffled_piece_puzzle(grid_size=(3, 3), image_size=620):
    """
    Decoupe une image en pieces, cache une piece pour la comparaison avec
    le dessin, melange les autres pieces, puis retourne du HTML interactif.
    """
    global saved_puzzle_img, saved_missing_piece, saved_puzzle_info

    image_path = get_random_image_path()
    original = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(original, (image_size, image_size), method=Image.Resampling.LANCZOS)

    rows, cols = grid_size
    piece_w = image_size // cols
    piece_h = image_size // rows

    pieces = []
    missing_index = random.randint(0, rows * cols - 1)
    for row in range(rows):
        for col in range(cols):
            correct_index = row * cols + col
            piece = image.crop(
                (
                    col * piece_w,
                    row * piece_h,
                    (col + 1) * piece_w,
                    (row + 1) * piece_h,
                )
            )
            pieces.append(
                {
                    "correct": correct_index,
                    "src": pil_to_data_url(piece),
                }
            )

    missing_piece = pieces[missing_index]
    saved_puzzle_img = np.array(image)
    saved_missing_piece = np.array(
        Image.open(io.BytesIO(base64.b64decode(missing_piece["src"].split(",", 1)[1]))).convert("RGB")
    )
    saved_puzzle_info = {
        "image_name": image_path.name,
        "emotion": image_path.parent.name,
        "rows": rows,
        "cols": cols,
        "missing_index": missing_index,
    }

    visible_pieces = [piece for piece in pieces if piece["correct"] != missing_index]
    shuffled = visible_pieces[:]
    while True:
        random.shuffle(shuffled)
        if any(piece["correct"] != index for index, piece in enumerate(shuffled)):
            break

    slot_contents = [""] * (rows * cols)
    fill_slots = [index for index in range(rows * cols) if index != missing_index]
    for slot_index, piece in zip(fill_slots, shuffled):
        slot_contents[slot_index] = f"""
                <img class="smart-puzzle-piece"
                     draggable="true"
                     data-piece="{piece['correct']}"
                     src="{piece['src']}"
                     alt="piece {piece['correct'] + 1}">
        """

    slots_html = []
    for slot_index, content in enumerate(slot_contents):
        missing_class = " smart-puzzle-missing-slot" if slot_index == missing_index else ""
        missing_attr = ' data-missing="1"' if slot_index == missing_index else ""
        slots_html.append(
            f"""
            <div class="smart-puzzle-slot{missing_class}" data-slot="{slot_index}"{missing_attr}>
                {content}
            </div>
            """
        )

    info = {
        "image_name": image_path.name,
        "emotion": image_path.parent.name,
        "rows": rows,
        "cols": cols,
    }

    puzzle_html = f"""
    <div class="smart-puzzle-wrap">
        <style>
            .smart-puzzle-wrap {{
                width: min(100%, 980px);
                margin: 0 auto;
                font-family: Arial, sans-serif;
            }}
            .smart-puzzle-shell {{
                display: grid;
                grid-template-columns: minmax(0, {image_size}px) minmax(220px, 1fr);
                gap: 22px;
                align-items: start;
            }}
            .smart-puzzle-board {{
                width: min(100%, {image_size}px);
                aspect-ratio: 1 / 1;
                display: grid;
                grid-template-columns: repeat({cols}, 1fr);
                grid-template-rows: repeat({rows}, 1fr);
                border: 3px solid #1b1b1b;
                background: #f6f6f6;
                box-shadow: 0 16px 42px rgba(0, 0, 0, 0.14);
            }}
            .smart-puzzle-slot {{
                min-width: 0;
                min-height: 0;
                border: 1px solid rgba(0, 0, 0, 0.28);
                background: #fff;
            }}
            .smart-puzzle-slot.drag-over {{
                outline: 3px solid #3498db;
                outline-offset: -3px;
            }}
            .smart-puzzle-missing-slot {{
                background:
                    repeating-linear-gradient(
                        45deg,
                        #f7f2d8,
                        #f7f2d8 12px,
                        #e6d98f 12px,
                        #e6d98f 24px
                    );
            }}
            .smart-puzzle-missing-slot::after {{
                content: "";
                display: block;
                width: 100%;
                height: 100%;
                border: 3px dashed rgba(0, 0, 0, 0.35);
                box-sizing: border-box;
            }}
            .smart-puzzle-piece {{
                display: block;
                width: 100%;
                height: 100%;
                object-fit: cover;
                cursor: grab;
                user-select: none;
            }}
            .smart-puzzle-piece:active {{
                cursor: grabbing;
            }}
            .smart-puzzle-side {{
                display: flex;
                flex-direction: column;
                gap: 14px;
            }}
            .smart-puzzle-actions {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }}
            .smart-puzzle-actions button {{
                border: 0;
                background: #1f6feb;
                color: #fff;
                border-radius: 8px;
                padding: 13px 14px;
                cursor: pointer;
                font-weight: 700;
                font-size: 15px;
                min-height: 46px;
            }}
            .smart-puzzle-actions button:hover {{
                filter: brightness(0.95);
            }}
            .smart-puzzle-actions .shuffle-smart-puzzle {{
                background: #2e7d32;
            }}
            .smart-puzzle-status {{
                min-height: 86px;
                padding: 16px;
                border: 2px solid #1b1b1b;
                border-radius: 8px;
                background: #fff7cc;
                color: #1f1f1f;
                font-size: 18px;
                line-height: 1.35;
                font-weight: 700;
                display: flex;
                align-items: center;
            }}
            @media (max-width: 760px) {{
                .smart-puzzle-shell {{
                    grid-template-columns: 1fr;
                }}
                .smart-puzzle-actions {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        <div class="smart-puzzle-shell">
            <div class="smart-puzzle-board" data-rows="{rows}" data-cols="{cols}">
                {''.join(slots_html)}
            </div>
            <div class="smart-puzzle-side">
                <div class="smart-puzzle-status">Deplace les pieces melangees. La case hachuree est la piece a dessiner.</div>
                <div class="smart-puzzle-actions">
                    <button type="button" class="check-smart-puzzle">Verifier</button>
                    <button type="button" class="shuffle-smart-puzzle">Melanger</button>
                </div>
            </div>
        </div>
    </div>
    """

    return puzzle_html, f"Puzzle genere : {info['emotion']} - {info['image_name']}"
