# train_gpt.py
# Long-run GPT training with periodic checkpointing. Use playground.py to sample.
import time

from NimbleML.losses import CrossEntropyLoss
from NimbleML.optimizers import Adam
from NimbleML.utils.np_backend import device, using_gpu

from NimbleML.data.text import batch_sequences
from gpt_common import (
    CHECKPOINT_DIR,
    LOG_EVERY,
    SAVE_EVERY_MINUTES,
    batches_per_epoch,
    build_model,
    checkpoint_exists,
    count_parameters,
    learning_rate_at_step,
    load_corpus_ids,
    load_training_checkpoint,
    model_config,
    save_training_checkpoint,
    train_step,
)


def print_training_banner(cfg, num_tokens, num_params, training_state):
    batches = batches_per_epoch(num_tokens, cfg["batch_size"], cfg["seq_len"])
    print(f"Device: {device} (gpu={using_gpu})")
    print(f"Corpus tokens: {num_tokens:,} | ~{batches:,} batches/epoch")
    print(
        f"Model: d={cfg['d_model']} L={cfg['num_layers']} H={cfg['num_heads']} "
        f"seq={cfg['seq_len']} vocab={cfg['vocab_size']} | params={num_params:,}"
    )
    print(
        f"LR: {cfg['learning_rate']:.2e} warmup={cfg['warmup_steps']} "
        f"cosine restart every {cfg['cosine_restart_every']} steps | grad clip {cfg['grad_clip_norm']}"
    )
    print(f"Checkpoint dir: {CHECKPOINT_DIR} | save every {SAVE_EVERY_MINUTES} min")
    if training_state.get("global_step", 0) > 0:
        print(
            f"Resuming: step={training_state['global_step']:,} "
            f"epoch={training_state['epoch']} "
            f"ema_loss={training_state.get('ema_loss', 0):.4f}"
        )
    print("Train until Ctrl+C. Progress auto-saves — run playground.py anytime.\n")


def main():
    ids, tokenizer = load_corpus_ids()
    cfg = model_config(tokenizer.vocab_size)

    if checkpoint_exists():
        print(f"Found checkpoint at {CHECKPOINT_DIR} — resuming.")
        model, optimizer, cfg, training_state = load_training_checkpoint()
    else:
        print("No checkpoint found — training from scratch.")
        model = build_model(cfg)
        optimizer = Adam(model.parameters(), learning_rate=cfg["learning_rate"])
        training_state = {"global_step": 0, "epoch": 0, "ema_loss": None}

    num_params = count_parameters(model)
    print_training_banner(cfg, len(ids), num_params, training_state)

    criterion = CrossEntropyLoss()
    global_step = training_state["global_step"]
    epoch = training_state["epoch"]
    ema_loss = training_state.get("ema_loss")
    ema_beta = 0.98

    batches_epoch = batches_per_epoch(len(ids), cfg["batch_size"], cfg["seq_len"])
    train_start = time.time()
    last_save = time.time()

    try:
        while True:
            epoch += 1
            epoch_loss = 0.0
            epoch_steps = 0

            for inputs, targets in batch_sequences(ids, cfg["batch_size"], cfg["seq_len"]):
                global_step += 1
                epoch_steps += 1
                optimizer.learning_rate = learning_rate_at_step(global_step, cfg)

                loss = train_step(
                    model,
                    criterion,
                    optimizer,
                    inputs,
                    targets,
                    cfg["grad_clip_norm"],
                )
                epoch_loss += loss
                ema_loss = loss if ema_loss is None else ema_beta * ema_loss + (1 - ema_beta) * loss

                if LOG_EVERY > 0 and global_step % LOG_EVERY == 0:
                    elapsed = time.time() - train_start
                    avg = epoch_loss / epoch_steps
                    print(
                        f"step {global_step:>7,} | epoch {epoch} | batch {epoch_steps}/{batches_epoch} "
                        f"| loss {avg:.4f} | ema {ema_loss:.4f} | lr {optimizer.learning_rate:.2e} "
                        f"| {elapsed / 3600:.2f}h"
                    )

                now = time.time()
                if (now - last_save) >= SAVE_EVERY_MINUTES * 60:
                    state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "ema_loss": ema_loss,
                    }
                    save_training_checkpoint(model, optimizer, cfg, state)
                    last_save = now
                    print(f"  [saved checkpoint @ step {global_step:,}]")

            if epoch_steps > 0:
                print(f"epoch {epoch} done | avg loss {epoch_loss / epoch_steps:.4f} | {epoch_steps} batches")

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        state = {
            "global_step": global_step,
            "epoch": epoch,
            "ema_loss": ema_loss,
        }
        save_training_checkpoint(model, optimizer, cfg, state)
        elapsed = time.time() - train_start
        print(
            f"Saved. step={global_step:,} epoch={epoch} ema_loss={ema_loss:.4f} "
            f"session={elapsed / 3600:.2f}h. Run playground.py to try the model."
        )


if __name__ == "__main__":
    main()
