import gradio as gr

from smartpuzzleai import puzzle


PUZZLE_JS = """
function initSmartPuzzle() {
    document.querySelectorAll('.smart-puzzle-board:not([data-ready])').forEach((board) => {
        board.dataset.ready = '1';
        const wrap = board.closest('.smart-puzzle-wrap');
        const status = wrap.querySelector('.smart-puzzle-status');
        const checkButton = wrap.querySelector('.check-smart-puzzle');
        const shuffleButton = wrap.querySelector('.shuffle-smart-puzzle');
        let draggedPiece = null;
        let moves = 0;

        function pieces() {
            return Array.from(board.querySelectorAll('.smart-puzzle-piece'));
        }

        function slots() {
            return Array.from(board.querySelectorAll('.smart-puzzle-slot'));
        }

        function setStatus(text) {
            status.textContent = text;
        }

        function attachPiece(piece) {
            piece.addEventListener('dragstart', (event) => {
                draggedPiece = piece;
                event.dataTransfer.effectAllowed = 'move';
                event.dataTransfer.setData('text/plain', piece.dataset.piece);
            });
            piece.addEventListener('dragend', () => {
                draggedPiece = null;
                slots().forEach((slot) => slot.classList.remove('drag-over'));
            });
        }

        pieces().forEach(attachPiece);

        slots().forEach((slot) => {
            slot.addEventListener('dragover', (event) => {
                if (slot.dataset.missing === '1') {
                    return;
                }
                event.preventDefault();
                slot.classList.add('drag-over');
            });
            slot.addEventListener('dragleave', () => {
                slot.classList.remove('drag-over');
            });
            slot.addEventListener('drop', (event) => {
                event.preventDefault();
                slot.classList.remove('drag-over');
                if (slot.dataset.missing === '1') {
                    return;
                }
                if (!draggedPiece) {
                    return;
                }

                const fromSlot = draggedPiece.parentElement;
                const targetPiece = slot.querySelector('.smart-puzzle-piece');
                if (targetPiece && targetPiece !== draggedPiece) {
                    fromSlot.appendChild(targetPiece);
                }
                slot.appendChild(draggedPiece);
                moves += 1;
                setStatus(`Mouvement ${moves}. Continue.`);
            });
        });

        checkButton.addEventListener('click', () => {
            const correct = slots().every((slot, index) => {
                if (slot.dataset.missing === '1') {
                    return slot.querySelector('.smart-puzzle-piece') === null;
                }
                const piece = slot.querySelector('.smart-puzzle-piece');
                return piece && Number(piece.dataset.piece) === index;
            });
            if (correct) {
                setStatus(`Bravo ! Puzzle reussi en ${moves} mouvements.`);
            } else {
                setStatus('Pas encore. Regarde les formes et continue.');
            }
        });

        shuffleButton.addEventListener('click', () => {
            const allPieces = pieces();
            for (let i = allPieces.length - 1; i > 0; i -= 1) {
                const j = Math.floor(Math.random() * (i + 1));
                [allPieces[i], allPieces[j]] = [allPieces[j], allPieces[i]];
            }
            slots().forEach((slot, index) => {
                slot.innerHTML = '';
                if (slot.dataset.missing !== '1' && allPieces.length) {
                    slot.appendChild(allPieces.shift());
                }
            });
            moves = 0;
            setStatus('Pieces melangees. A toi de jouer.');
        });
    });
}

new MutationObserver(initSmartPuzzle).observe(document.body, { childList: true, subtree: true });
setTimeout(initSmartPuzzle, 200);
"""


with gr.Blocks(title="Smart Puzzle AI") as demo:
    gr.Markdown("# Smart Puzzle AI")
    gr.Markdown("Reconstruisez l'image avec les pieces melangees, puis dessinez la piece manquante.")

    def new_puzzle_board():
        puzzle_html, _info = puzzle.create_shuffled_piece_puzzle()
        return puzzle_html

    with gr.Row():
        with gr.Column(scale=3):
            puzzle_board = gr.HTML()
            new_btn = gr.Button("Nouveau Puzzle", variant="primary")

        with gr.Column(scale=2):
            drawing_pad = gr.ImageEditor(
                label="Dessinez ici la piece manquante",
                type="numpy",
            )
            compare_btn = gr.Button("Verifier mon dessin", variant="secondary")
            result_box = gr.Textbox(label="Resultat", interactive=False)

    demo.load(fn=new_puzzle_board, outputs=[puzzle_board])
    new_btn.click(fn=new_puzzle_board, outputs=[puzzle_board])
    compare_btn.click(
        fn=puzzle.compare_with_missing_piece,
        inputs=[drawing_pad],
        outputs=[result_box],
    )


if __name__ == "__main__":
    demo.launch(js=PUZZLE_JS)
