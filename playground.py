# playground.py
# Load the latest training checkpoint and interactively generate text.
from gpt_common import (
    CHECKPOINT_DIR,
    checkpoint_exists,
    generation_settings,
    load_training_checkpoint,
    run_playground,
)
from NimbleML.data.tokenizer import BPETokenizer
from NimbleML.utils.np_backend import device, using_gpu


def main():
    if not checkpoint_exists():
        print(f"No checkpoint found at {CHECKPOINT_DIR}")
        print("Run train_gpt.py first (or wait for the first auto-save).")
        return

    print(f"Loading checkpoint from {CHECKPOINT_DIR} (device={device}, gpu={using_gpu})...")
    model, _optimizer, cfg, training_state = load_training_checkpoint()
    tokenizer = BPETokenizer.load(cfg["tokenizer_path"])

    step = training_state.get("global_step", "?")
    ema = training_state.get("ema_loss")
    ema_str = f"{ema:.4f}" if ema is not None else "n/a"
    print(f"Checkpoint: step={step:,} epoch={training_state.get('epoch', '?')} ema_loss={ema_str}\n")

    run_playground(model, tokenizer, cfg)


if __name__ == "__main__":
    main()
