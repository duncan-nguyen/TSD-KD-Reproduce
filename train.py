import os
import sys

import torch
import torch.distributed as dist
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import GKDConfig

from DistillTrainer import DistillTrainer

# Load environment variables from .env file
load_dotenv()

# Login to Hugging Face Hub
hf_token = os.getenv("HF_TOKEN")
hf_repo_id = os.getenv("HF_REPO_ID")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in environment variables.")

beta = float(sys.argv[1])
lmbda = float(sys.argv[2])
threshold = float(sys.argv[3])
model_name = (
    sys.argv[4]
    if len(sys.argv) > 4 and sys.argv[4]
    else os.getenv("STUDENT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
)

import torch._dynamo

torch._dynamo.config.disable = True


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


local_rank = int(os.environ.get("LOCAL_RANK", "0"))
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

teacher_model_name = os.getenv("TEACHER_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")
attn = "sdpa"
# The model to optimise
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation=attn,
    torch_dtype=torch.bfloat16,
    pad_token_id=tokenizer.pad_token_id,
    trust_remote_code=True,
)  # .to(f"cuda:{local_rank}")

# The teacher model to calculate the KL divergence against
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    attn_implementation=attn,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)  # .to(f"cuda:{local_rank}")
model.resize_token_embeddings(teacher_model.lm_head.weight.shape[0])

print(model.lm_head.weight.shape)
print(teacher_model.lm_head.weight.shape)

assert model.lm_head.weight.shape[0] == teacher_model.lm_head.weight.shape[0]

ds = load_dataset("Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen")["train"].train_test_split(
    test_size=0.01
)


def add_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


train_dataset = ds["train"].map(add_messages).remove_columns(["prompt"])
eval_dataset = ds["test"].map(add_messages).remove_columns(["prompt"])

fsdp_config = {
    "limit_all_gathers": True,
    "forward_prefetch": True,
    "backward_prefetch": "backward_pre",
}
training_args = GKDConfig(
    output_dir="tsd-kd-Qwen2.5-1.5B-Instruct",
    logging_steps=10,
    num_train_epochs=3,
    warmup_ratio=0.1,
    per_device_eval_batch_size=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    learning_rate=5e-6,
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    lr_scheduler_type="cosine",
    bf16=True,
    max_length=1024,
    save_total_limit=3,
    report_to="wandb",
    lmbda=lmbda,
    beta=beta,
    temperature=1.0,
    push_to_hub=hf_repo_id is not None,
    hub_model_id=hf_repo_id,
    hub_strategy="every_save",
)


trainer = DistillTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    token_entropy_percentile_threshold=threshold,
)
trainer.train()
