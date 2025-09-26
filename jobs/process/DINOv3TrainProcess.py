import os
from collections import OrderedDict
from typing import Optional
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm

from jobs.process.BaseTrainProcess import BaseTrainProcess
from toolkit.config_modules import ModelConfig, TrainConfig, SaveConfig, OxenConfig
from toolkit.models.dinov3 import DINOv3
from toolkit.optimizer import get_optimizer
from toolkit.scheduler import get_lr_scheduler
from toolkit.train_tools import get_torch_dtype
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

# Import Oxen integration (with try/except for optional dependency)
try:
    from toolkit.oxen_experiment import AIToolkitOxenExperiment
    from toolkit.oxen_logger import AIToolkitOxenLogger
    OXEN_AVAILABLE = True
except ImportError:
    OXEN_AVAILABLE = False
    AIToolkitOxenExperiment = None
    AIToolkitOxenLogger = None


class DINOv3TrainProcess(BaseTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)

        self.accelerator = get_accelerator()
        self.device = self.accelerator.device

        self.model_config = ModelConfig(**self.get_conf('model', {}))
        self.train_config = TrainConfig(**self.get_conf('train', {}))
        self.save_config = SaveConfig(**self.get_conf('save', {}))

        # Initialize Oxen experiment tracking if enabled
        self.oxen_config = OxenConfig(**self.get_conf('oxen', {}))
        self.oxen_experiment = None
        self.oxen_logger = None

        self.dinov3_model: Optional[DINOv3] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.step_num = 0
        self.epoch_num = 0

        # Training state
        self.dtype = get_torch_dtype(self.train_config.dtype)

    def run(self):
        print_acc("Starting DINOv3 segmentation training...")

        self.setup_model()
        train_dataloader = self.setup_dataloader()
        self.setup_optimizer()
        self.setup_scheduler()

        # Prepare for distributed training
        self.dinov3_model, self.optimizer, train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.dinov3_model, self.optimizer, train_dataloader, self.lr_scheduler
        )

        # Initialize Oxen experiment tracking if enabled
        if self.oxen_config.enabled and OXEN_AVAILABLE and self.oxen_experiment is None:
            if self.accelerator.is_main_process:
                print_acc("Initializing Oxen experiment tracking...")
                try:
                    # Get model name from config
                    model_name = self.model_config.name_or_path or "dinov3-vitl16-pretrain-lvd1689m"

                    # Initialize experiment
                    self.oxen_experiment = AIToolkitOxenExperiment(
                        repo_id=self.oxen_config.repo_id,
                        base_model_name=model_name,
                        fine_tuned_model_name=self.config.get('name', 'dinov3_segmentation'),
                        output_dir_base=self.oxen_config.output_dir_base,
                        is_main_process=True,
                        host=self.oxen_config.host,
                        scheme=self.oxen_config.scheme,
                    )

                    # Initialize logger
                    self.oxen_logger = AIToolkitOxenLogger(
                        experiment=self.oxen_experiment,
                        is_main_process=True,
                        fine_tune_id=self.oxen_config.fine_tune_id,
                    )

                    print_acc(f"Oxen experiment initialized: {self.oxen_experiment.name}")

                except Exception as e:
                    print_acc(f"Warning: Failed to initialize Oxen experiment: {e}")
                    self.oxen_experiment = None
                    self.oxen_logger = None

            # Broadcast experiment details to other processes if using distributed training
            if hasattr(self.accelerator, 'num_processes') and self.accelerator.num_processes > 1:
                if self.accelerator.is_main_process:
                    details = self.oxen_experiment.get_details_for_broadcast() if self.oxen_experiment else {}
                else:
                    details = {}

                # Broadcast details to all processes
                details = self.accelerator.broadcast_object_list([details])[0]

                if not self.accelerator.is_main_process and details:
                    # Initialize experiment on non-main processes
                    self.oxen_experiment = AIToolkitOxenExperiment(
                        repo_id=self.oxen_config.repo_id,
                        base_model_name=details.get('base_model_name', 'dinov3'),
                        fine_tuned_model_name=details.get('fine_tuned_model_name', 'dinov3_segmentation'),
                        output_dir_base=self.oxen_config.output_dir_base,
                        is_main_process=False,
                        host=self.oxen_config.host,
                        scheme=self.oxen_config.scheme,
                    )

                    self.oxen_experiment.update_from_broadcast(details)

        self.training_loop(train_dataloader)

        # Finalize Oxen experiment if enabled
        if self.accelerator.is_main_process:
            if self.oxen_logger and self.oxen_config.enabled:
                try:
                    print_acc("Finalizing Oxen experiment...")
                    self.oxen_logger.finalize_experiment(self.save_root)
                    print_acc("Oxen experiment finalized successfully")
                except Exception as e:
                    print_acc(f"Warning: Failed to finalize Oxen experiment: {e}")

        print_acc("Training completed!")

    def setup_model(self):
        print_acc("Initializing DINOv3 model...")

        self.dinov3_model = DINOv3(
            config=self.model_config,
            device=self.device
        )

        total_params = sum(p.numel() for p in self.dinov3_model.dinov3_model.parameters()) + sum(p.numel() for p in self.dinov3_model.segmentation_decoder.parameters())
        trainable_params = sum(p.numel() for p in self.dinov3_model.segmentation_decoder.parameters() if p.requires_grad)

        print_acc(f"Total parameters: {total_params:,}")
        print_acc(f"Trainable parameters: {trainable_params:,}")
        print_acc(f"Frozen parameters: {total_params - trainable_params:,}")

    def setup_dataloader(self):
        print_acc("Setting up data loader...")

        # TODO: improve this / use the AI Toolkit dataloader.
        # Base dataloader has too much stuff specific to diffusion models,
        # so this is just a simple dataloader that "works"
        from torch.utils.data import Dataset, DataLoader
        import os
        import glob
        from PIL import Image
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        import torch

        PATCH_SIZE = 16
        IMAGE_SIZE = 768

        # quantization filter for the given patch size
        patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
        patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

        # image resize transform to dimensions divisible by patch size
        def resize_transform(
            mask_image: Image.Image,
            image_size: int = IMAGE_SIZE,
            patch_size: int = PATCH_SIZE,
        ) -> torch.Tensor:
            w, h = mask_image.size
            # print_acc(w, h)
            h_patches = int(image_size / patch_size)
            w_patches = int((w * image_size) / (h * patch_size))
            # print_acc(w_patches * patch_size, h_patches * patch_size)
            return TF.to_tensor(TF.resize(mask_image, [h_patches * patch_size, w_patches * patch_size]))

        class SimpleSegmentationDataset(Dataset):
            def __init__(self, image_dir, label_dir, target_size=512, num_classes=7):
                self.image_dir = image_dir
                self.label_dir = label_dir
                self.target_size = target_size
                self.num_classes = num_classes

                # Find all images and their corresponding masks
                self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
                self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, "*.jpeg"))))
                self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, "*.png"))))
                self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.jpg")))
                self.label_paths.extend(sorted(glob.glob(os.path.join(label_dir, "*.jpeg"))))
                self.label_paths.extend(sorted(glob.glob(os.path.join(label_dir, "*.png"))))

                # Filter to only include images that have corresponding masks
                valid_paths = []
                for img_path in self.image_paths:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    lbl_path = os.path.join(label_dir, f"{base_name}.png")
                    if os.path.exists(lbl_path):
                        valid_paths.append(img_path)

                self.image_paths = valid_paths
                print_acc(f"Found {len(self.image_paths)} valid image-label pairs")

                # Image transforms (normalize for DINOv3)
                self.transform = T.Compose([
                    # T.Resize((target_size, target_size)),
                    # T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                # Mask transform
                # self.mask_transform = T.Compose([
                #     T.Resize((target_size, target_size), interpolation=T.InterpolationMode.NEAREST),
                # ])

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                # Load image
                # print_acc(idx)
                img_path = self.image_paths[idx]
                image = Image.open(img_path).convert('RGB')
                # print_acc(image.size)
                image = resize_transform(image, image_size=self.target_size)
                image = self.transform(image)
                # print_acc(image.shape)

                # Load label image - support both alpha channel and grayscale
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(self.label_dir, f"{base_name}.png")
                # label_path = os.path.join(self.image_dir, f"{base_name}_seg.png")

                # Load label image in original format first
                label_pil = Image.open(label_path)

                # Check if we have alpha channel and this is binary segmentation
                if label_pil.mode == 'RGBA' and self.num_classes == 2:
                    # print_acc(f"Using alpha channel for binary segmentation: {label_path}")
                    # Extract alpha channel for binary segmentation
                    label = label_pil.split()[-1]
                    label_resized = resize_transform(label, image_size=self.target_size)
                    label_quantized = patch_quant_filter(label_resized).squeeze().detach()
                    label = (label_quantized > 0).long()
                else:
                    # Use standard grayscale label loading
                    label = label_pil.convert('L')
                    label = resize_transform(label, image_size=self.target_size).long()
                # print_acc("Image/label sizes:")
                # print_acc(image.shape)
                # print_acc(label.shape)

                return {
                    'images': image,
                    'segmentation_masks': label
                }

        datasets_config = self.get_conf('datasets', [])
        if not datasets_config:
            raise ValueError("No datasets configuration found")

        dataset_path = datasets_config[0]['folder_path']
        label_path = datasets_config[0]['label_path']
        target_size = datasets_config[0].get('resolution', [512])[0]
        print_acc(f"Target size: {target_size}")

        num_classes = self.model_config.num_classes
        dataset = SimpleSegmentationDataset(dataset_path, label_path, target_size, num_classes)
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        return dataloader

    def setup_optimizer(self):
        print_acc("Setting up optimizer...")

        # Only optimize decoder parameters (backbone is frozen)
        trainable_params = list(self.dinov3_model.get_trainable_parameters())

        optimizer_params = {}
        if 'weight_decay' in self.train_config.optimizer_params:
            optimizer_params['weight_decay'] = self.train_config.optimizer_params["weight_decay"]
        else:
            optimizer_params['weight_decay'] = 0.05

        print_acc(f"optimizer_params['weight_decay'] == {optimizer_params['weight_decay']}")

        self.optimizer = get_optimizer(
            trainable_params,
            optimizer_type=self.train_config.optimizer,
            learning_rate=self.train_config.lr,
            optimizer_params=optimizer_params
        )

    def setup_scheduler(self):
        if (hasattr(self.train_config, 'lr_scheduler') and 
            self.train_config.lr_scheduler is not None and 
            self.train_config.lr_scheduler.strip() != ""):
            scheduler_name = self.train_config.lr_scheduler
            print_acc(f"Setting up {scheduler_name} scheduler...")

            scheduler_kwargs = {}
            if scheduler_name in ['cosine', 'cosine_with_restarts']:
                scheduler_kwargs['total_iters'] = self.train_config.steps
            elif scheduler_name == 'constant_with_warmup':
                scheduler_kwargs['num_warmup_steps'] = getattr(self.train_config, 'warmup_steps', 500)
                scheduler_kwargs['total_iters'] = self.train_config.steps

            self.lr_scheduler = get_lr_scheduler(
                name=scheduler_name,
                optimizer=self.optimizer,
                **scheduler_kwargs
            )
        else:
            print_acc("No learning rate scheduler configured, using constant learning rate")
            self.lr_scheduler = None

    def training_loop(self, train_dataloader):
        print_acc(f"Starting training for {self.train_config.steps} steps...")

        # Set decoder to training mode (backbone stays frozen in eval mode)
        self.dinov3_model.segmentation_decoder.train()

        progress_bar = tqdm(total=self.train_config.steps, desc="Training")

        while self.step_num < self.train_config.steps:
            for batch in train_dataloader:
                if self.step_num >= self.train_config.steps:
                    break

                class BatchWrapper:
                    def __init__(self, batch_dict):
                        self.images = batch_dict['images']
                        self.segmentation_masks = batch_dict['segmentation_masks']

                batch_wrapper = BatchWrapper(batch)

                # Training step
                loss = self.training_step(batch_wrapper)

                # Update progress
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})
                progress_bar.update(1)

                # Log to Oxen if enabled
                if self.oxen_logger and self.oxen_config.enabled:
                    try:
                        # Prepare metrics for Oxen
                        learning_rate = self.optimizer.param_groups[0]['lr']
                        oxen_metrics = {
                            'step': self.step_num,
                            'learning_rate': learning_rate,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'loss': loss.item(),
                            'epoch': self.epoch_num,
                        }

                        self.oxen_logger.log_metrics(oxen_metrics, self.step_num)
                    except Exception as e:
                        print_acc(f"Warning: Failed to log metrics to Oxen: {e}")

                # Save checkpoint
                if self.step_num % self.save_config.save_every == 0 and self.step_num > 0:
                    self.save_checkpoint()

                self.step_num += 1

        progress_bar.close()

        # Save final checkpoint
        self.save_checkpoint()

    def training_step(self, batch: DataLoaderBatchDTO):
        logits = self.dinov3_model.forward(batch)

        loss = self.compute_loss(batch, logits)

        self.accelerator.backward(loss)

        if hasattr(self.train_config, 'max_grad_norm') and self.train_config.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.dinov3_model.segmentation_decoder.parameters(), self.train_config.max_grad_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()

        return loss

    def compute_loss(self, batch, logits: torch.Tensor):
        targets = batch.segmentation_masks.to(self.device).long()

        if targets.shape[-2:] != logits.shape[-2:]:
            print_acc(f"[WARNING] Resizing because {targets.shape[-2:]} != {logits.shape[-2:]}")
            targets = F.interpolate(
                targets.float().unsqueeze(1),
                size=logits.shape[-2:],
                mode='nearest'
            ).squeeze(1).long()

        # unique_classes = np.unique(targets.cpu())
        # print_acc(unique_classes)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, targets, ignore_index=255)

        return loss

    def save_checkpoint(self):
        if not self.accelerator.is_main_process:
            return

        print_acc(f"Saving checkpoint at step {self.step_num}")

        checkpoint_dir = os.path.join(self.save_root, f"checkpoint_{self.step_num}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_path = os.path.join(checkpoint_dir, "dinov3_decoder.pth")
        self.dinov3_model.save_model(model_path)

        training_state = {
            'step': self.step_num,
            'epoch': self.epoch_num,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        if self.lr_scheduler is not None:
            training_state['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pth"))

        print_acc(f"Checkpoint saved to {checkpoint_dir}")

        # Log checkpoint to Oxen if enabled
        if self.oxen_logger and self.oxen_config.enabled:
            try:
                self.oxen_logger.save_checkpoint(checkpoint_dir, self.step_num)
            except Exception as e:
                print_acc(f"Warning: Failed to log checkpoint to Oxen: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        print_acc(f"Loading checkpoint from {checkpoint_path}")

        # Load model
        model_path = os.path.join(checkpoint_path, "dinov3_decoder.pth")
        if os.path.exists(model_path):
            self.dinov3_model.load_model(model_path)

        training_state_path = os.path.join(checkpoint_path, "training_state.pth")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            self.step_num = training_state['step']
            self.epoch_num = training_state['epoch']
            self.optimizer.load_state_dict(training_state['optimizer_state_dict'])

            if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in training_state:
                self.lr_scheduler.load_state_dict(training_state['lr_scheduler_state_dict'])

        print_acc(f"Checkpoint loaded. Resuming from step {self.step_num}")
