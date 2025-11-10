from datasets import load_dataset
from SERL_code.trl.trainer.SERL_trainer import SERLTrainer
from SERL_code.trl.trainer.SERL_config import SERLConfig

pre_model = "Qwen3-8B" # your pre_model path
data_files_path = "SERL_code/data/train_Ultrafeedback.jsonl"
dataset = load_dataset(path = 'json', data_files = data_files_path, split="train")

training_args = SERLConfig(output_dir="Qwen3-8B-SERL",
                            per_device_train_batch_size=2,
                            N_num_generations_actor = 4,
                            K_num_generations_judge = 4,
                            num_train_epochs=5.0,
                            gradient_accumulation_steps=4,
                            logging_steps=1,
                            max_prompt_length=4096,
                            max_prompt_length_p2=4096,
                            max_completion_length=1024,
                            max_completion_length_p2=2048,
                            do_train=True,
                            save_strategy="steps",
                            save_steps=4,
                            log_completions=True,
                            beta=0.0,
                            temperature=0.9,
                            warmup_ratio=0.05,
                            gradient_checkpointing=True,
                            learning_rate=1e-6,
                            lr_scheduler_type="cosine")
 
trainer = SERLTrainer(
    model=pre_model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()