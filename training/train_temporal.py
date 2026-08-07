"""Train and export the MobileNetV4 temporal cursor-hotspot detector."""

import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.temporal_dataset import OUTPUT_STRIDE, TemporalCursorDataset
from training.temporal_model import TemporalCursorExport, TemporalCursorNet, temporal_cursor_loss


def decode_batch(heatmap_logits, offsets):
    batch, _, height, width = heatmap_logits.shape
    scores, indices = heatmap_logits.sigmoid().flatten(1).max(dim=1)
    ys = torch.div(indices, width, rounding_mode="floor")
    xs = indices % width
    rows = torch.arange(batch, device=offsets.device)
    dx = offsets[rows, 0, ys, xs]
    dy = offsets[rows, 1, ys, xs]
    points = torch.stack([(xs + dx) * OUTPUT_STRIDE, (ys + dy) * OUTPUT_STRIDE], dim=1)
    return scores, points


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    errors, true_positive, predicted_positive, actual_positive = [], 0, 0, 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        heatmap, offsets = model(inputs)
        scores, points = decode_batch(heatmap, offsets)
        visible = batch["visible"].to(device)
        predicted = scores >= 0.30
        distances = (points - batch["position"].to(device)).norm(dim=1)
        errors.extend(distances[visible].cpu().tolist())
        true_positive += (predicted & visible & (distances <= 8)).sum().item()
        predicted_positive += predicted.sum().item()
        actual_positive += visible.sum().item()
    precision = true_positive / max(1, predicted_positive)
    recall = true_positive / max(1, actual_positive)
    mean_error = sum(errors) / max(1, len(errors))
    within4 = sum(error <= 4 for error in errors) / max(1, len(errors))
    within8 = sum(error <= 8 for error in errors) / max(1, len(errors))
    return {"precision@8": precision, "recall@8": recall, "mean_error": mean_error, "within4": within4, "within8": within8}


def main():
    device_name = os.getenv("CURSOR_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    batch_size = int(os.getenv("CURSOR_BATCH", "16" if device.type == "mps" else "32"))
    workers = int(os.getenv("CURSOR_WORKERS", "2"))
    epochs = int(os.getenv("CURSOR_EPOCHS", "20"))
    negative_rate = float(os.getenv("CURSOR_NEGATIVE_RATE", "0.05"))
    train_set = TemporalCursorDataset("train", int(os.getenv("CURSOR_TRAIN_SAMPLES", "10000")), negative_rate, dynamic=True)
    val_set = TemporalCursorDataset("val", int(os.getenv("CURSOR_VAL_SAMPLES", "1000")), negative_rate, dynamic=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")

    model = TemporalCursorNet(pretrained=os.getenv("CURSOR_PRETRAINED", "1") != "0").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.getenv("CURSOR_LR", "0.001")), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(os.getenv("CURSOR_OUTPUT", "training/runs/mobilenetv4-temporal"))
    output.mkdir(parents=True, exist_ok=True)
    best_recall = -1.0

    print(f"device={device} batch={batch_size} workers={workers} train={len(train_set)} val={len(val_set)}")
    for epoch in range(epochs):
        train_set.set_epoch(epoch)
        model.train()
        running = [0.0, 0.0, 0.0]
        started = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            inputs = batch["input"].to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                heatmap, offsets = model(inputs)
                loss, heatmap_loss, offset_loss = temporal_cursor_loss(heatmap, offsets, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            values = (loss.item(), heatmap_loss.item(), offset_loss.item())
            running = [total + value for total, value in zip(running, values)]
            if step == 1 or step % 50 == 0 or step == len(train_loader):
                averages = [value / step for value in running]
                samples_per_second = step * batch_size / (time.perf_counter() - started)
                print(f"epoch {epoch + 1}/{epochs} step {step}/{len(train_loader)} loss={averages[0]:.4f} heatmap={averages[1]:.4f} offset={averages[2]:.4f} samples/s={samples_per_second:.1f}", flush=True)
        scheduler.step()
        metrics = validate(model, val_loader, device)
        print("validation", " ".join(f"{key}={value:.4f}" for key, value in metrics.items()), flush=True)
        checkpoint = {"model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics}
        torch.save(checkpoint, output / "last.pt")
        if metrics["recall@8"] > best_recall:
            best_recall = metrics["recall@8"]
            torch.save(checkpoint, output / "best.pt")

    checkpoint = torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dummy = torch.zeros(1, 3, 640, 640, device=device)
    torch.onnx.export(
        TemporalCursorExport(model), dummy, output / "mobilenetv4-temporal.onnx",
        input_names=["frames"], output_names=["heatmap", "offset"], opset_version=17,
        dynamo=False,
    )
    print(f"exported {output / 'mobilenetv4-temporal.onnx'}")


if __name__ == "__main__":
    main()
