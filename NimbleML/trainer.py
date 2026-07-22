"""Training loop utilities (Trainer + fit())."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
from NimbleML.utils.clip_grad import clip_grad_norm_
from NimbleML.utils.saveload import load_checkpoint, save_checkpoint
from NimbleML.data.dataset import PADDED_LABEL


@dataclass
class TrainerState:
    """Simple state snapshot for callbacks and return values."""
    epoch: int = 0
    global_step: int = 0


class TrainerCallback:
    """Callback base class for Trainer hooks.

    Users can subclass this or pass any object with the same method names.
    """

    def on_step_end(self, trainer: "Trainer", *, epoch: int, step: int, loss: Any) -> None:  # pragma: no cover
        pass

    def on_epoch_end(self, trainer: "Trainer", *, epoch: int) -> None:  # pragma: no cover
        pass

    def on_checkpoint(
        self,
        trainer: "Trainer",
        *,
        checkpoint_path: str,
        epoch: int,
        step: int,
    ) -> None:  # pragma: no cover
        pass


class Trainer:
    """A minimal epoch/step training loop.

    Step order:
        zero_grad -> scheduler(step) -> forward/compute_loss -> backward -> clip -> optimizer.step
    """

    def __init__(
        self,
        model,
        optimizer,
        *,
        scheduler=None,
        max_grad_norm: float | None = None,
        callbacks: Iterable[Any] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.callbacks = list(callbacks) if callbacks is not None else []
        self.state = TrainerState()

    def _call_callbacks(self, method_name: str, **kwargs: Any) -> None:
        for cb in self.callbacks:
            fn = getattr(cb, method_name, None)
            if fn is None:
                continue
            fn(self, **kwargs)

    def _extract_loss_from_batch(self, batch, *, ignore_index=None):
        if not (isinstance(batch, tuple) and len(batch) == 2):
            raise ValueError(
                "Trainer expects batches as (input_ids, labels). "
                f"Got batch type={type(batch).__name__}, value={batch!r}"
            )
        inputs, labels = batch
        if ignore_index is None:
            # SequenceLMDataset pads targets with PADDED_LABEL; TokenLM ids are >= 0.
            ignore_index = PADDED_LABEL
        if not hasattr(self.model, "compute_loss"):
            raise ValueError("Model must implement compute_loss(inputs, labels) for Trainer.fit().")
        return self.model.compute_loss(inputs, labels, ignore_index=ignore_index)

    def train_step(self, batch, *, ignore_index: int | None = None) -> Any:
        """Run one training step for a single batch."""
        # Put model into training mode for layers like dropout.
        if hasattr(self.model, "train"):
            self.model.train()

        if self.scheduler is not None:
            # Scheduler APIs in NimbleML use last_epoch as a generic step index.
            # Calling with an explicit `epoch=` makes behavior deterministic for resume/tests.
            self.scheduler.step(epoch=self.state.global_step)

        self.optimizer.zero_grad(set_to_none=True)
        loss = self._extract_loss_from_batch(batch, ignore_index=ignore_index)
        loss.backward()

        if self.max_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        return loss

    def fit(
        self,
        dataloader,
        *,
        epochs: int,
        start_epoch: int = 0,
        start_step: int = 0,
        max_steps: int | None = None,
        resume_from: str | None = None,
        checkpoint_path: str | None = None,
        checkpoint_every_steps: int | None = None,
        ignore_index: int | None = None,
    ):
        """Train for `epochs` epochs using an epoch/step loop."""

        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        if resume_from is not None:
            ckpt = load_checkpoint(
                resume_from,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            ckpt_step = ckpt.get("step") or 0
            self.state.global_step = int(ckpt_step)
            steps_per_epoch = len(dataloader) if hasattr(dataloader, "__len__") else None
            if steps_per_epoch:
                start_epoch = int(self.state.global_step // steps_per_epoch)
                start_step = int(self.state.global_step % steps_per_epoch)
            else:
                start_epoch = 0
                start_step = int(self.state.global_step)
        else:
            self.state.global_step = int(start_step)
            self.state.epoch = int(start_epoch)

        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        end_step = None if max_steps is None else self.state.global_step + int(max_steps)

        for epoch in range(start_epoch, start_epoch + int(epochs)):
            self.state.epoch = epoch
            self.model.train()

            batch_i = 0
            for batch in dataloader:
                # Skip already-trained batches when resuming mid-epoch.
                if resume_from is not None and epoch == start_epoch and batch_i < start_step:
                    batch_i += 1
                    continue

                if end_step is not None and self.state.global_step >= end_step:
                    break

                loss = self.train_step(batch, ignore_index=ignore_index)
                self.state.global_step += 1
                step = self.state.global_step

                self._call_callbacks("on_step_end", epoch=epoch, step=step, loss=loss)

                if (
                    checkpoint_path is not None
                    and checkpoint_every_steps is not None
                    and checkpoint_every_steps > 0
                    and (step % checkpoint_every_steps == 0)
                ):
                    ckpt_file = self._format_checkpoint_path(checkpoint_path, epoch=epoch, step=step)
                    save_checkpoint(
                        ckpt_file,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        step=step,
                        extra={"epoch": epoch, "step": step},
                    )
                    self._call_callbacks(
                        "on_checkpoint",
                        checkpoint_path=ckpt_file,
                        epoch=epoch,
                        step=step,
                    )

                batch_i += 1

            self._call_callbacks("on_epoch_end", epoch=epoch)

            if end_step is not None and self.state.global_step >= end_step:
                break

        return {"epoch": self.state.epoch, "global_step": self.state.global_step}

    @staticmethod
    def _format_checkpoint_path(checkpoint_path: str, *, epoch: int, step: int) -> str:
        # Allow patterns like "ckpt_step{step}.npz" or "ckpt_epoch{epoch}_step{step}.npz".
        try:
            return checkpoint_path.format(epoch=epoch, step=step)
        except Exception:
            # If the user provided a plain filename without placeholders, just return it.
            return checkpoint_path

