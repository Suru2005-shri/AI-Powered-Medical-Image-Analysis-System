"""
main.py
=======
CLI entry point for the AI Medical Image Analysis project.

Usage
-----
  python main.py --mode download          # download Kaggle dataset
  python main.py --mode train             # train the model
  python main.py --mode evaluate          # evaluate on test set
  python main.py --mode predict --image path/to/xray.jpg   # single image
  python main.py --mode batch   --dir  data/sample_images  # all images in folder
  python main.py --mode demo              # end-to-end demo on sample images
  python main.py --mode all              # run everything sequentially
"""

import argparse
import os
import sys

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Medical Image Analysis – CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["download", "train", "evaluate", "predict", "batch", "demo", "all"],
        required=True,
        help="Operation to perform.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image (used with --mode predict).",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory of images (used with --mode batch).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override path to saved model (.keras file).",
    )
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        help="Skip Phase 2 fine-tuning during training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Download ───────────────────────────────────────────────────────────────
    if args.mode in ("download", "all"):
        print("\n[STEP] Downloading dataset …")
        from src.data_loader import download_dataset
        download_dataset()

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.mode in ("train", "all"):
        print("\n[STEP] Training model …")
        from src.train import train
        histories = train(run_fine_tuning=not args.no_finetune)
        print("[DONE] Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if args.mode in ("evaluate", "all"):
        print("\n[STEP] Evaluating model …")
        from src.evaluate import evaluate
        model_path = args.model or None
        metrics = evaluate(model_path=model_path) if model_path else evaluate()
        print(f"\n[DONE] Test Accuracy : {metrics['test_accuracy']*100:.2f}%")
        print(f"       F1-score (macro) : {metrics['f1_macro']:.4f}")

    # ── Single image predict ───────────────────────────────────────────────────
    if args.mode == "predict":
        if not args.image:
            print("[ERROR] Provide --image <path> for predict mode.")
            sys.exit(1)
        from src.predict import predict_single
        from src.config import SAVED_MODEL_PATH
        model_path = args.model or SAVED_MODEL_PATH
        result = predict_single(args.image, model_path=model_path)
        print(f"\nResult: {result}")

    # ── Batch predict ──────────────────────────────────────────────────────────
    if args.mode == "batch":
        image_dir = args.dir or "data/sample_images"
        from src.predict import predict_batch
        from src.config import SAVED_MODEL_PATH
        model_path = args.model or SAVED_MODEL_PATH
        predict_batch(image_dir, model_path=model_path)

    # ── Demo ──────────────────────────────────────────────────────────────────
    if args.mode == "demo":
        _run_demo(args)


def _run_demo(args):
    """
    End-to-end demo that works even without downloading the full dataset.

    It:
      1. Generates synthetic X-ray-like images (so you can run without Kaggle)
      2. Shows preprocessing stages
      3. Builds and summarises the model (without training)
      4. Saves architecture diagram to outputs/
    """
    print("\n" + "="*60)
    print("  DEMO MODE – synthetic images, no training required")
    print("="*60)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.config import SAMPLE_DIR, GRAPHS_DIR, OUTPUTS_DIR
    from src.model import build_model, get_model_summary

    # ── 1. Generate synthetic 'X-ray-like' images ─────────────────────────────
    print("\n[STEP 1] Generating synthetic chest X-ray images …")
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    img_paths = []

    for i in range(6):
        h, w = 512, 512
        # Create a dark background with rib-like ellipse patterns
        img = np.zeros((h, w), dtype=np.uint8)

        # Lung fields (bright ovals)
        for cx, cy, rx, ry in [(180, 280, 100, 150), (332, 280, 100, 150)]:
            Y, X = np.ogrid[:h, :w]
            mask = ((X - cx)**2 / rx**2 + (Y - cy)**2 / ry**2) <= 1
            img[mask] = np.random.randint(100, 180)

        # Rib-like arcs
        for rib_y in range(180, 400, 30):
            cv_pts = np.linspace(100, 420, 50).astype(int)
            rib_curve = (rib_y + 20 * np.sin(np.linspace(0, np.pi, 50))).astype(int)
            for x, y in zip(cv_pts, rib_curve):
                if 0 <= x < w and 0 <= y < h:
                    img[max(0, y-2):y+2, max(0, x-1):x+1] = 200

        # Add noise
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Save as RGB
        img_rgb = np.stack([img, img, img], axis=-1)
        path    = os.path.join(SAMPLE_DIR, f"synthetic_xray_{i+1:02d}.png")
        plt.imsave(path, img_rgb)
        img_paths.append(path)

    print(f"[INFO] Saved {len(img_paths)} synthetic images to {SAMPLE_DIR}")

    # ── 2. Preprocessing stages ────────────────────────────────────────────────
    print("\n[STEP 2] Visualising preprocessing pipeline …")
    from src.visualize import plot_preprocessing_stages
    plot_preprocessing_stages(img_paths[0])

    # ── 3. Dataset sample grid (synthetic) ────────────────────────────────────
    print("\n[STEP 3] Creating sample image grid …")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Synthetic Chest X-Ray Samples (Demo)", fontsize=14, fontweight="bold")
    for ax, path in zip(axes.flatten(), img_paths):
        img = plt.imread(path)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(os.path.basename(path), fontsize=9)
    grid_path = os.path.join(GRAPHS_DIR, "demo_samples.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Sample grid → {grid_path}")

    # ── 4. Model summary ───────────────────────────────────────────────────────
    print("\n[STEP 4] Building model architecture …")
    model = build_model(fine_tune_at=0)
    get_model_summary(model)

    # ── 5. Save architecture diagram ──────────────────────────────────────────
    try:
        tf_plot_path = os.path.join(OUTPUTS_DIR, "model_architecture.png")
        import tensorflow as tf
        tf.keras.utils.plot_model(
            model,
            to_file=tf_plot_path,
            show_shapes=True,
            show_layer_names=True,
            dpi=96,
        )
        print(f"[INFO] Architecture diagram → {tf_plot_path}")
    except Exception as e:
        print(f"[WARN] Could not save architecture diagram: {e}")

    print("\n[DONE] Demo complete! Check the outputs/ folder for visualisations.")
    print("       To train for real, run:  python main.py --mode train")


if __name__ == "__main__":
    main()
