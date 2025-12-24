# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
import signal
import yaml

import torch
import torch.distributed as dist
import torch.utils.data
import torchvision.transforms.functional as F
from megatron.core import parallel_state
import wandb

from cosmos_predict1.utils import callback, distributed, ema, log, misc
from cosmos_predict1.utils.checkpointer import Checkpointer
from cosmos_predict1.utils.lazy_config import LazyConfig, instantiate
from cosmos_predict1.utils.model import Model


class Trainer:
    """The base trainer class.

    All trainers should inherit Trainer. It contains the basic functionality for model training
    (particularly suited for large-scale training), including data parallel (DDP/FSDP), model weight average (EMA),
    mixed-precision training (fp16/bf16).

    Attributes:
        checkpointer (Checkpointer): checkpointer object to save/load model weights and optimizer states.
        training_timer (misc.Timer): Timer object to time code blocks and functions.
    """

    def __init__(self, config):
        """Constructor of the trainer.

        Args:
            config (Config): The config object for the codebase.
        """
        super().__init__()
        self.config = config
        # Set up the distributed computing environment.
        with misc.timer("init_distributed"):
            distributed.init()
            # Set up parallel states.
            if hasattr(config.model, "context_parallel_size"):
                if config.model_parallel.context_parallel_size > 1:
                    raise ValueError(
                        "Both config.model.context_parallel_size and config.model_parallel.context_parallel_size are set. "
                        "config.model.context_parallel_size is deprecated. Please only set config.model_parallel.context_parallel_size."
                    )
                else:
                    log.critical(
                        "Using deprecated config.model.context_parallel_size. Please use config.model_parallel.context_parallel_size instead."
                    )
                    config.model_parallel.context_parallel_size = config.model.context_parallel_size
            parallel_state.initialize_model_parallel(
                pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                context_parallel_size=config.model_parallel.context_parallel_size,
            )
            # `config.model_parallel.sequence_parallel` is a bool that indicates whether to use sequence parallelism.
            # It is not part of the original `parallel_state` API, so we need to set it manually.
            parallel_state.sequence_parallel = config.model_parallel.sequence_parallel
            if parallel_state.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # Create the local job directory, save the config file, and pipe to a local log.
        if distributed.is_rank0():
            os.makedirs(config.job.path_local, exist_ok=True)
            # Save the config as .pkl for reproducibility.
            LazyConfig.save_pkl(config, f"{config.job.path_local}/config.pkl")
            # Save the config as .yaml for reading or parsing experiment hyperparameters.
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
            # Read the saved yaml file and convert it to a dictionary
            with open(f"{config.job.path_local}/config.yaml", 'r') as f:
                self.config_dict = yaml.safe_load(f)
        dist.barrier()
        log.init_loguru_file(f"{config.job.path_local}/stdout.log")
        if distributed.is_rank0():
            # Print important environment variables and the effective config.
            log.info("Config:\n" + config.pretty_print(use_color=True))
        misc.print_environ_variables(["TORCH_HOME", "OUTPUT_ROOT"])
        # Set the random seed. If multi-GPU, different ranks are set with different seeds.
        misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
        # Initialize cuDNN.
        torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
        torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
        # Floating-point precision settings.
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        # Initialize the callback functions.
        self.callbacks = callback.CallBackGroup(config=config, trainer=self)
        # Initialize the model checkpointer.
        if config.checkpoint.type is None:
            self.checkpointer = Checkpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        else:
            self.checkpointer: Checkpointer = instantiate(
                config.checkpoint.type, config.checkpoint, config.job, callbacks=self.callbacks
            )
        # Initialize the timer for speed benchmarking.
        self.training_timer = misc.TrainingTimer()
        # Send a TimeoutError if a training step takes over timeout_period seconds.
        signal.signal(signal.SIGALRM, functools.partial(misc.timeout_handler, config.trainer.timeout_period))  # type: ignore

    def train(
        self,
        model: Model,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """The training function.

        Args:
            model (Model): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Initialize wandb for tracking experiments
        if distributed.is_rank0():
            print(self.config)
            wandb.init(
                project=self.config.job.project,
                group=self.config.job.group,
                name=self.config.job.name,
                config=self.config_dict
            )
            
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        model.on_train_start(self.config.trainer.memory_format)

        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()
        # Load the model checkpoint and get the starting iteration number.
        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        log.info("Starting training...")
        self.callbacks.on_train_start(model, iteration=iteration)
        # Initial validation.
        if self.config.trainer.run_validation and iteration == 0:
            self.validate(model, dataloader_val, iteration=iteration)
        _end_training = False
        while True:
            dataloader_train_iter = iter(dataloader_train)
            while True:
                self.callbacks.on_before_dataloading(iteration)
                with self.training_timer("dataloader_train"):
                    try:
                        data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        break
                self.callbacks.on_after_dataloading(iteration)
                # If max_iter is reached, exit the training loop.
                if iteration >= self.config.trainer.max_iter:
                    _end_training = True
                    break
                # Move all tensors in the data batch to GPU device.
                data_batch = misc.to(data_batch, device="cuda")
                # The actual training step.
                self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                if not model.training:
                    model_ddp.train()
                assert model_ddp.training, "model_ddp is not in training mode."
                assert model.training, "model is not in training mode."
                output_batch, loss, grad_accum_iter = self.training_step(
                    model_ddp,
                    optimizer,
                    scheduler,
                    grad_scaler,
                    data_batch,
                    iteration=iteration,
                    grad_accum_iter=grad_accum_iter,
                )
                # Log loss to wandb
                if distributed.is_rank0() and iteration % self.config.trainer.logging_iter == 0:
                    print(f"Iteration={iteration:6d}, loss={loss:.4f}")
                    wandb.log({"loss": loss.item()}, step=iteration)
                # Do the following when an actual optimizer (update) step has been made.
                iteration += 1
                # Save checkpoint.
                if iteration % self.config.checkpoint.save_iter == 0:
                    self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
                # Validation.
                if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                    input_example, output_example, _ = self.validate(model, dataloader_val, iteration=iteration)
                    if distributed.is_rank0():
                        self.visualize(input_example, output_example, iteration)  
                # This iteration is successful; reset the timeout signal.
                signal.alarm(self.config.trainer.timeout_period)
            if _end_training:
                break
        log.success("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        
        # Finish wandb run
        if distributed.is_rank0():
            wandb.finish()
            
        self.callbacks.on_app_end()

    def training_step(
        self,
        model_ddp: torch.nn.Module | distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        """The training step.

        Args:
            model_ddp (torch.nn.Module | distributed.DistributedDataParallel): The model with a DDP wrapper or, the bare
              module, depending on whether distributed training is enabled or not.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).
            iteration (int): Current iteration number.
            grad_accum_iter (int): Number of gradient accumulation iterations.

        Returns:
            output (dict[str, torch.Tensor]): The model output from the training data batch (dictionary of tensors).
            loss (torch.Tensor): The total loss of the training data batch.
        """
        # Only let DDP sync gradient at the last iteration of the gradient accumulation window
        with distributed.ddp_sync_grad(model_ddp, grad_accum_iter == self.config.trainer.grad_accum_iter - 1):
            with self.training_timer("forward"):
                output_batch, loss = model_ddp.training_step(data, iteration)
            self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
            with self.training_timer("backward"):
                loss_scaled = grad_scaler.scale(loss / self.config.trainer.grad_accum_iter)
                loss_scaled.backward()
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_after_backward()
                else:
                    model_ddp.on_after_backward()
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)
        grad_accum_iter += 1
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            with self.training_timer("optimizer_step"):
                self.callbacks.on_before_optimizer_step(
                    model_ddp, optimizer, scheduler, grad_scaler, iteration=iteration
                )
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                self.callbacks.on_before_zero_grad(model_ddp, optimizer, scheduler, iteration=iteration)
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                else:
                    model_ddp.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                optimizer.zero_grad(set_to_none=True)
            grad_accum_iter = 0
        return output_batch, loss, grad_accum_iter

    @torch.no_grad()
    def validate(self, model: Model, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0) -> None:
        """Validate on the full validation dataset.

        Args:
            model (Model): The PyTorch model.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
            iteration (int): Current iteration number.
        """
        self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
        model.eval()
        # Evaluate on the full validation set.
        with ema.ema_scope(model, enabled=model.config.ema.enabled):
            for val_iter, data_batch in enumerate(dataloader_val):
                if self.config.trainer.max_val_iter is not None and val_iter >= self.config.trainer.max_val_iter:
                    break
                data_batch = misc.to(data_batch, device="cuda")
                self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
                output_batch, loss = model.validation_step(data_batch, iteration)
                if val_iter == 0:
                    input_example = data_batch["video"] # (B, C, T, H, W)
                    output_example = output_batch["prediction"] # (B, C, T, H, W)
                    loss_example = loss
                self.callbacks.on_validation_step_end(model, data_batch, output_batch, loss, iteration=iteration)
        self.callbacks.on_validation_end(model, iteration=iteration)
        return input_example, output_example, loss_example
    
    @torch.no_grad()
    def visualize(self, input_tensor, output_tensor, iteration):
        """Visualize the input and output tensors.
        
        Args:
            input_tensor (torch.Tensor): The input tensor. Shape: (B, C, T, H, W)
            output_tensor (torch.Tensor): The output tensor. Shape: (B, C, T, H, W)
            iteration (int): Current iteration number.
        """
        import numpy as np
        import os
        
        # input_tensor, output_tensor: (B, C, T, H, W)
        is_image = input_tensor.shape[2] == 1  # T == 1 means single frame (image)
        num_examples = min(8, input_tensor.shape[0])
        
        # Normalize from [-1, 1] to [0, 1]
        def normalize(x):
            return ((x.detach().cpu().float() + 1) / 2).clamp(0, 1)
        
        input_vis = normalize(input_tensor[:num_examples])   # (N, C, T, H, W)
        output_vis = normalize(output_tensor[:num_examples]) # (N, C, T, H, W)
        
        if is_image:
            # Image visualization: T=1, squeeze temporal dimension
            for i in range(num_examples):
                # (C, 1, H, W) -> (C, H, W) -> (H, W, C)
                input_img = input_vis[i, :, 0].permute(1, 2, 0)
                output_img = output_vis[i, :, 0].permute(1, 2, 0)
                # Concatenate vertically for comparison
                img = torch.cat([input_img, output_img], dim=0)  # (2H, W, C)
                img = F.to_pil_image(img.permute(2, 0, 1))  # to_pil_image expects (C, H, W)
                wandb.log(
                    {f"visualization/image_{i}": wandb.Image(img, caption=f"Top: Input, Bottom: Output")},
                    step=iteration
                )
        else:
            # Video visualization: T > 1
            import imageio
            
            # Create output directory
            output_root = os.environ.get("OUTPUT_ROOT", ".")
            save_dir = os.path.join(output_root, "wandb_videos", f"iter_{iteration:09d}")
            os.makedirs(save_dir, exist_ok=True)
            
            for i in range(num_examples):
                # (C, T, H, W) -> (T, H, W, C)
                inp = input_vis[i].permute(1, 2, 3, 0)   # (T, H, W, C)
                out = output_vis[i].permute(1, 2, 3, 0) # (T, H, W, C)
                
                # Ensure same temporal length
                min_t = min(inp.shape[0], out.shape[0])
                inp, out = inp[:min_t], out[:min_t]
                
                # Concatenate side by side (along width)
                combined = torch.cat([inp, out], dim=2)  # (T, H, 2W, C)
                
                # Convert to uint8
                combined_np = (combined.numpy() * 255).clip(0, 255).astype(np.uint8)
                
                # Handle channel count (RGB only)
                if combined_np.shape[-1] > 3:
                    combined_np = combined_np[..., :3]
                
                # Save video
                video_path = os.path.join(save_dir, f"video_{i}.mp4")
                imageio.mimwrite(video_path, combined_np, fps=16, macro_block_size=None)
                
                # Log to wandb
                wandb.log(
                    {f"visualization/video_{i}": wandb.Video(
                        video_path, 
                        caption=f"Left: Input, Right: Output",
                        fps=8,
                        format="mp4"
                    )},
                    step=iteration
                )
