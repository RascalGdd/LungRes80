import os
import json
import numpy as np
import math
import sys
from contextlib import nullcontext
from typing import Iterable, Optional
import torch
from datasets.transforms.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from datetime import datetime
from scipy.special import softmax


def train_class_batch(
    model,
    samples,
    target,
    criterion,
    recent_phase_histories=None,
    current_time_seconds=None,
):
    if recent_phase_histories is None:
        outputs = model(samples)
    else:
        outputs = model(
            samples,
            recent_phase_histories=recent_phase_histories,
            current_time_seconds=current_time_seconds,
        )
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    if hasattr(optimizer, "loss_scale"):
        return optimizer.loss_scale
    if hasattr(optimizer, "cur_scale"):
        return optimizer.cur_scale
    # BF16 uses its full exponent range and therefore has no dynamic scaler.
    return 1.0


@torch.no_grad()
def collect_prediction_history_records(model, data_loader, device):
    """Run a checkpoint over the train set without updates for bank bootstrap."""
    model.eval()
    records = {"indices": [], "probabilities": []}
    metric_logger = utils.MetricLogger(delimiter="  ")
    for samples, _, ids, _ in metric_logger.log_every(
        data_loader, 100, "Bootstrap history bank:"
    ):
        samples = samples.to(device, non_blocking=True)
        samples = samples.to(dtype=next(model.parameters()).dtype)
        output = model(samples)
        records["indices"].extend(
            int(identifier.split("_", 1)[0]) for identifier in ids
        )
        records["probabilities"].append(
            output.float().softmax(dim=-1).cpu().numpy()
        )
    return records


def corrupt_phase_histories(histories, probability, num_classes=7):
    """Delete or replace one transition so history cannot become a shortcut."""
    corrupted = []
    for history in histories:
        history = list(history)
        if history and torch.rand(()) < probability:
            index = int(torch.randint(len(history), ()).item())
            if torch.rand(()) < 0.5:
                del history[index]
            else:
                offset = int(torch.randint(1, num_classes, ()).item())
                history[index] = (int(history[index]) + offset) % num_classes
        corrupted.append(history)
    return corrupted


def _evaluation_precision_context(model, videos):
    """Match evaluation precision without nesting autocast around DeepSpeed."""
    model_dtype = next(model.parameters()).dtype
    is_deepspeed = callable(getattr(model, "backward", None)) and callable(
        getattr(model, "step", None)
    )
    if is_deepspeed:
        return videos.to(dtype=model_dtype), nullcontext()
    use_autocast = model_dtype in (torch.float16, torch.bfloat16)
    context = torch.autocast(
        device_type="cuda",
        dtype=model_dtype if use_autocast else torch.float16,
        enabled=use_autocast,
    )
    return videos, context


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    prediction_history_active=False,
    history_dropout=0.0,
    history_corruption_prob=0.0,
    history_prediction_records=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, ids, metadata) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            break
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if (
            lr_schedule_values is not None
            or wd_schedule_values is not None
            and data_iter_step % update_freq == 0
        ):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        recent_phase_histories = None
        current_time_seconds = [
            int(identifier.strip().rsplit("_", 1)[1]) for identifier in ids
        ]
        if prediction_history_active:
            if "prediction_recent_phases" not in metadata:
                raise RuntimeError(
                    "Prediction-history training requires a populated history bank"
                )
            padded_histories = metadata["prediction_recent_phases"]
            history_counts = metadata["prediction_recent_phase_count"]
            padded_times = metadata.get("prediction_recent_phase_times")
            recent_phase_histories = []
            for row_index, (row, count) in enumerate(
                zip(padded_histories, history_counts)
            ):
                phases = row[: int(count)].tolist()
                if padded_times is None:
                    recent_phase_histories.append(phases)
                else:
                    times = padded_times[row_index, : int(count)].tolist()
                    recent_phase_histories.append(list(zip(phases, times)))
            if history_dropout > 0:
                keep = torch.rand(len(recent_phase_histories)) >= history_dropout
                recent_phase_histories = [
                    history if bool(use_history) else []
                    for history, use_history in zip(recent_phase_histories, keep)
                ]
            if history_corruption_prob > 0:
                recent_phase_histories = corrupt_phase_histories(
                    recent_phase_histories, history_corruption_prob
                )
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            # DeepSpeed has already converted registered model parameters to
            # its configured FP16/BF16 dtype. Match the input explicitly;
            # forcing FP16 here breaks native BF16 xLSTM training.
            samples = samples.to(dtype=next(model.parameters()).dtype)
            loss, output = train_class_batch(
                model,
                samples,
                targets,
                criterion,
                recent_phase_histories=recent_phase_histories,
                current_time_seconds=current_time_seconds,
            )
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model,
                    samples,
                    targets,
                    criterion,
                    recent_phase_histories=recent_phase_histories,
                    current_time_seconds=current_time_seconds,
                )

        loss_value = loss.item()
        if history_prediction_records is not None:
            probabilities = output.detach().float().softmax(dim=-1).cpu().numpy()
            history_prediction_records["indices"].extend(
                int(identifier.split("_", 1)[0]) for identifier in ids
            )
            history_prediction_records["probabilities"].append(probabilities)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(
    data_loader,
    model,
    device,
    prediction_file=None,
    split_name="Val",
    static_prediction_history=False,
):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"{split_name}:"

    # switch to evaluation mode
    model.eval()
    online_model = getattr(model, "module", model)
    use_online_inference = callable(getattr(online_model, "predict_online", None))
    if use_online_inference:
        online_model.reset_stream_state()
    raw_predictions = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        metadata = batch[3]

        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        videos, precision_context = _evaluation_precision_context(model, videos)
        with precision_context:
            if use_online_inference:
                video_ids = [identifier.strip().split("_", 2)[1] for identifier in ids]
                current_time_seconds = [
                    int(identifier.strip().rsplit("_", 1)[1])
                    for identifier in ids
                ]
                if static_prediction_history:
                    if "prediction_recent_phases" not in metadata:
                        raise RuntimeError(
                            "Static-history evaluation requires cached phase histories"
                        )
                    padded_histories = metadata["prediction_recent_phases"]
                    history_counts = metadata["prediction_recent_phase_count"]
                    padded_times = metadata.get("prediction_recent_phase_times")
                    recent_phase_histories = []
                    for row_index, (row, count) in enumerate(
                        zip(padded_histories, history_counts)
                    ):
                        phases = row[: int(count)].tolist()
                        if padded_times is None:
                            recent_phase_histories.append(phases)
                        else:
                            times = padded_times[
                                row_index, : int(count)
                            ].tolist()
                            recent_phase_histories.append(
                                list(zip(phases, times))
                            )
                    online_result = online_model.predict_online_with_histories(
                        videos,
                        video_ids,
                        recent_phase_histories,
                        current_time_seconds=current_time_seconds,
                    )
                else:
                    online_result = online_model.predict_online(
                        videos,
                        video_ids,
                        current_time_seconds=current_time_seconds,
                    )
                raw_logits = online_result["raw_logits"]
                output = online_result["smoothed_probabilities"].clamp_min(1e-12).log()
                loss = criterion(raw_logits, target)
            else:
                raw_logits = model(videos)
                output = raw_logits
                loss = criterion(raw_logits, target)

        if prediction_file is not None:
            logits = raw_logits.detach().float().cpu()
            raw_probabilities = logits.softmax(dim=-1)
            raw_predictions_ids = logits.argmax(dim=-1).tolist()
            if use_online_inference:
                smoothed_probabilities = online_result["smoothed_probabilities"].cpu()
                predictions = online_result["smoothed_predictions"].cpu().tolist()
            else:
                smoothed_probabilities = raw_probabilities
                predictions = raw_predictions_ids
            targets = target.detach().cpu().tolist()
            for identifier, prediction, raw_prediction, label, scores, raw_probs, smooth_probs in zip(
                ids,
                predictions,
                raw_predictions_ids,
                targets,
                logits.tolist(),
                raw_probabilities.tolist(),
                smoothed_probabilities.tolist(),
            ):
                _, video_id, frame_id = identifier.strip().split("_", 2)
                raw_predictions.append(
                    {
                        "video_id": video_id,
                        "frame_id": int(frame_id),
                        "target": int(label),
                        "prediction": int(prediction),
                        "raw_prediction": int(raw_prediction),
                        "logits": scores,
                        "raw_probabilities": raw_probs,
                        "smoothed_probabilities": smooth_probs,
                    }
                )
        
        # for i in range(output.size(0)):
        #     if flags[i]:
        #         if target[i] == 0:
        #             output.data[i] = torch.tensor([1, 0, 0, 0, 0, 0, 0])
        #         elif target[i] == 1:
        #             output.data[i] = torch.tensor([0, 1, 0, 0, 0, 0, 0])
        #         elif target[i] == 2:
        #             output.data[i] = torch.tensor([0, 0, 1, 0, 0, 0, 0])

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    if prediction_file is not None:
        os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
        temporary_file = prediction_file + ".tmp"
        with open(temporary_file, "w", encoding="utf-8") as handle:
            for row in raw_predictions:
                handle.write(json.dumps(row) + "\n")
        os.replace(temporary_file, prediction_file)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_phase_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        flags = batch[3]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        videos, precision_context = _evaluation_precision_context(model, videos)
        with precision_context:
            output = model(videos)  
            # Output: N,7
            # Target: N,
            loss = criterion(output, target)

        for i in range(output.size(0)):
            unique_id, video_id, frame_id = ids[i].strip().split('_')
            # if flags[i]:
            #     if target[i] == 0:
            #         output.data[i] = torch.tensor([1, 0, 0, 0, 0, 0, 0])
            #     elif target[i] == 1:
            #         output.data[i] = torch.tensor([0, 1, 0, 0, 0, 0, 0])
            #     elif target[i] == 2:
            #         output.data[i] = torch.tensor([0, 0, 1, 0, 0, 0, 0])

            string = "{} {} {} {} {}\n".format(
                unique_id,
                video_id,
                frame_id,
                str(output.data[i].cpu().numpy().tolist()),
                str(int(target[i].cpu().numpy())),
            )
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        # os.mknod(file)  # 用于创建一个指定文件名的文件系统节点，暂时无权限
        open(file, 'a').close()
    with open(file, "w") as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + ".txt")
        print("Merge File %d/%d: %s" % (x+1, num_tasks, file))
        lines = open(file, "r").readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split("[")[0]
            label = line.split("]")[1].split(" ")[1]
            data = np.fromstring(
                line.split("[")[1].split("]")[0], dtype=float, sep=","
            )

            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0

            dict_feats[name].append(data)
            dict_label[name] = label
    print("Computing final results")
    
    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
        # 在这里存一下合并的输出，多GPU测试之后保留输出，用于评测更细致的指标
    from multiprocessing import Pool

    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)
    return final_top1 * 100, final_top5 * 100


def compute_video(lst):
    _, _, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
