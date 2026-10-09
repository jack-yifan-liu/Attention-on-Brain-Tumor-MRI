"""Shared classification epochs, metrics and checkpoint-selection helpers."""

import csv

import torch


def classification_metrics(targets, predictions, num_classes):
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Metrics require nonempty, aligned target/prediction lists")
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, predicted in zip(targets, predictions):
        if not (0 <= truth < num_classes and 0 <= predicted < num_classes):
            raise ValueError("Class index outside the saved mapping")
        matrix[truth][predicted] += 1
    f1 = []
    for label in range(num_classes):
        tp = matrix[label][label]
        fp = sum(row[label] for row in matrix) - tp
        fn = sum(matrix[label]) - tp
        denominator = 2 * tp + fp + fn
        # Macro-F1 always averages over every mapped class; a zero denominator counts as 0.
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {"accuracy": sum(matrix[i][i] for i in range(num_classes)) / len(targets),
            "macro_f1": sum(f1) / num_classes, "per_class_f1": f1,
            "confusion_matrix": matrix, "samples": len(targets)}


def run_epoch(model, loader, criterion, device, num_classes, *, optimizer=None,
              validate_updates=False):
    training = optimizer is not None
    model.train(training)
    total_loss, targets, predicted, records = 0.0, [], [], []
    checked_updates = {}
    for batch_number, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        if (images.dtype != torch.float32 or labels.dtype != torch.int64
                or images.ndim != 4 or images.shape[1] != 3
                or labels.ndim != 1 or labels.shape[0] != images.shape[0]
                or not torch.isfinite(images).all()
                or labels.min() < 0 or labels.max() >= num_classes):
            raise ValueError("Invalid input tensor, dtype or class range")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            if logits.shape != (len(labels), num_classes) or not torch.isfinite(logits).all():
                raise ValueError("Invalid model output shape or non-finite logits")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            if training:
                loss.backward()
                if validate_updates:
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad and (parameter.grad is None
                                                       or not torch.isfinite(parameter.grad).all()):
                            raise ValueError(f"Missing or non-finite gradient: {name}")
                sentinels = {}
                if validate_updates and batch_number == 0:
                    sentinels = {name: parameter.detach().clone()
                                 for name, parameter in model.named_parameters()
                                 if name in ("conv1.weight", "fc.weight") or name.startswith("avgpool.0.")}
                optimizer.step()
                if validate_updates:
                    for name, parameter in model.named_parameters():
                        if not torch.isfinite(parameter).all():
                            raise ValueError(f"Non-finite updated parameter: {name}")
                        if name in sentinels:
                            checked_updates[name] = not torch.equal(sentinels[name], parameter.detach())
                    if sentinels and not all(checked_updates.values()):
                        raise ValueError("Expected backbone/head/attention parameter update did not occur")
        probabilities = logits.detach().softmax(dim=1).cpu()
        guesses = probabilities.argmax(dim=1).tolist()
        truths = labels.detach().cpu().tolist()
        total_loss += loss.item() * len(labels)
        targets.extend(truths)
        predicted.extend(guesses)
        for index, (truth, guess) in enumerate(zip(truths, guesses)):
            records.append({"sample_id": batch["sample_id"][index],
                            "patient_id": batch["patient_id"][index],
                            "patient_group_id": batch["patient_group_id"][index],
                            "true_label": truth, "predicted_label": guess,
                            **{f"prob_{label}": float(probabilities[index, label])
                               for label in range(num_classes)}})
    metrics = classification_metrics(targets, predicted, num_classes)
    metrics["loss"] = total_loss / len(targets)
    if training:
        metrics["parameter_updates_checked"] = checked_updates
    return metrics, records


def is_better_checkpoint(candidate, best, *, tie_break):
    if tie_break not in ("lower_validation_loss", "earlier_epoch"):
        raise ValueError(f"Unknown checkpoint tie rule: {tie_break}")
    if best is None or candidate["macro_f1"] > best["macro_f1"]:
        return True
    if candidate["macro_f1"] == best["macro_f1"] and tie_break == "lower_validation_loss":
        return candidate["loss"] < best["loss"]
    return False


def write_predictions(path, rows, num_classes):
    fields = ["epoch", "sample_id", "patient_id", "patient_group_id", "true_label", "predicted_label",
              *(f"prob_{label}" for label in range(num_classes))]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
