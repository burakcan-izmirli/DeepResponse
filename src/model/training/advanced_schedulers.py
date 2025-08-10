"""
Advanced learning rate schedulers for improved training performance.
"""
import tensorflow as tf
import numpy as np
import math

class CosineAnnealingWithWarmup(tf.keras.callbacks.Callback):
    """
    Cosine annealing learning rate scheduler with warmup.
    
    Better convergence than step-based schedulers, especially for
    transformer-based models like SELFormer.
    """
    def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs, steps_per_epoch):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        
    def on_batch_begin(self, batch, logs=None):
        current_step = int(self.model.optimizer.iterations.numpy())
        
        if current_step < self.warmup_steps and self.warmup_steps > 0:
            # Linear warmup
            lr = self.max_lr * (current_step / self.warmup_steps)
        elif self.total_steps > self.warmup_steps:
            # Cosine annealing
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            lr = self.max_lr
            
        # Ensure lr is a float and within reasonable bounds
        lr = max(float(lr), 1e-8)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

class OneCycleLR(tf.keras.callbacks.Callback):
    """
    OneCycle learning rate scheduler - often provides faster convergence.
    
    Based on the "Super-convergence" paper by Leslie Smith.
    Particularly effective for transformer models.
    """
    def __init__(self, max_lr, total_steps, pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        self.step_up_size = int(total_steps * pct_start)
        self.step_down_size = total_steps - self.step_up_size
        
    def on_batch_begin(self, batch, logs=None):
        current_step = int(self.model.optimizer.iterations.numpy())
        
        if current_step <= self.step_up_size and self.step_up_size > 0:
            # Increase phase
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (current_step / self.step_up_size)
        else:
            # Decrease phase
            step_down = current_step - self.step_up_size
            if self.anneal_strategy == 'cos' and self.step_down_size > 0:
                lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * step_down / self.step_down_size))
            elif self.step_down_size > 0:  # linear
                lr = self.max_lr - (self.max_lr - self.min_lr) * (step_down / self.step_down_size)
            else:
                lr = self.min_lr
                
        # Ensure lr is a float and within reasonable bounds
        lr = max(float(lr), 1e-8)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

class PerformanceMonitor(tf.keras.callbacks.Callback):
    """
    Enhanced performance monitoring with early stopping and best model saving.
    """
    def __init__(self, patience=15, min_delta=1e-5, restore_best_weights=True):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss is None:
            return
            
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')

def get_advanced_callbacks(strategy_creator, checkpoint_path, steps_per_epoch, comet_logger=None):
    """
    Create advanced callbacks for improved training performance.
    
    Args:
        strategy_creator: Strategy creator instance
        checkpoint_path: Path to save model checkpoints
        steps_per_epoch: Number of steps per epoch
        comet_logger: Comet experiment logger
    
    Returns:
        List of advanced callbacks
    """
    args = strategy_creator.args
    max_lr = args.learning_rate
    min_lr = max_lr / 100  # Minimum learning rate
    total_epochs = args.epoch
    
    # Auto-select scheduler based on model configuration
    if args.selformer_trainable_layers > 0:
        scheduler_type = 'onecycle'  # Fine-tuning scenario
    else:
        scheduler_type = 'cosine_warmup'  # Frozen features scenario
    
    callbacks = []
    
    # Advanced learning rate scheduling
    if scheduler_type == 'onecycle':
        total_steps = total_epochs * steps_per_epoch
        callbacks.append(OneCycleLR(
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4
        ))
    else:  # Default to cosine_warmup
        warmup_epochs = 5  # Fixed default value
        warmup_epochs = min(warmup_epochs, total_epochs // 10)  # Cap at 10% of epochs
        callbacks.append(CosineAnnealingWithWarmup(
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch
        ))
    
    # Enhanced model checkpointing
    callbacks.extend([
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path.replace('.h5', '_best_val.h5').replace('.keras', '_best_val.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path.replace('.h5', '_best_r2.h5').replace('.keras', '_best_r2.h5'),
            save_best_only=True,
            monitor='val_r2_score',
            mode='max',
            save_weights_only=True,
            verbose=1
        ),
        
        # Enhanced performance monitoring
        PerformanceMonitor(
            patience=15,  # Fixed default value
            min_delta=1e-5,
            restore_best_weights=True
        )
    ])
    
    # Add backup traditional scheduler
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=8,
        min_lr=min_lr,
        verbose=1,
        cooldown=3
    ))
    
    return callbacks

def get_scheduler_recommendation(model_type, dataset_size, selformer_layers):
    """
    Get recommended scheduler based on model configuration.
    
    Args:
        model_type: Type of model ('hybrid', 'selformer', 'cnn')
        dataset_size: Size of dataset ('small', 'medium', 'large')
        selformer_layers: Number of trainable SELFormer layers
    
    Returns:
        dict: Scheduler recommendation
    """
    if selformer_layers > 0:
        # Fine-tuning scenario
        return {
            'type': 'onecycle',
            'reason': 'OneCycle scheduler works well for fine-tuning scenarios',
            'max_lr_multiplier': 1.0
        }
    elif selformer_layers == 0:
        # Frozen feature extraction
        return {
            'type': 'cosine_warmup',
            'reason': 'Cosine annealing with warmup for frozen feature extraction',
            'max_lr_multiplier': 1.5
        }
    else:
        # Full training
        return {
            'type': 'cosine_warmup',
            'reason': 'Cosine annealing for full model training',
            'max_lr_multiplier': 1.0
        }
