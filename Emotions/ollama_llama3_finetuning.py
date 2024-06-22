
# # Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip uninstall unsloth -y
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@nightly"
# !pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

from unsloth import FastLanguageModel
from unsloth import to_sharegpt, standardize_sharegpt
from unsloth import apply_chat_template
from unsloth import is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

print('=== Loading Model ===')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

print('=== Formatting dataset ===')

dataset = load_dataset("sebdg/fine-tune-emotions", split = "train")
dataset = to_sharegpt(
    dataset,
    merged_prompt = "{instruction}[[\nYour input is:\n{input}]]",
    conversation_extension = 3,
)
dataset = standardize_sharegpt(dataset)

chat_template = """Below are some tasks!

### Instruction:
{INPUT}

### Response:
{OUTPUT}

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
)

# print('=== Writing model file ===')

# with open("./supply_chain_llama3.Modelfile", "w", encoding="utf-8") as f:
#     f.write(tokenizer._ollama_modelfile)


print('=== Training the model ===')
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 500,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

print('=== Training ===')

trainer_stats = trainer.train()
print(trainer_stats)

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


print('=== Saving model ===')

model.save_pretrained("emotional_llama_lora_model") # Local saving
tokenizer.save_pretrained("emotional_llama_lora_tokenizer") # Local saving

# model.push_to_hub("sebdg/supply_chain_llama3_lora_model",  token = "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") # Online saving

# # Merge to 16bit
# model.save_pretrained_merged("supply_chain_llama3_merged_16bit", tokenizer, save_method = "merged_16bit",)

# # Merge to 4bit
# model.save_pretrained_merged("supply_chain_llama3_merged_4bit", tokenizer, save_method = "merged_4bit",)

#model.save_pretrained_merged("emotional_llama_merged_lora", tokenizer, save_method = "lora",)

# Save to 8bit Q8_0
#model.save_pretrained_gguf("emotional_llama_q8", tokenizer,)

# # Save to 16bit GGUF
# model.save_pretrained_gguf("supply_chain_llama3_f16", tokenizer, quantization_method = "f16")

# Save to q4_k_m GGUF
model.save_pretrained_gguf("emotional_llama_q4_k_m", tokenizer, quantization_method = "q4_k_m")
model.push_to_hub_gguf("sebdg/emotional_llama_q4_k_m", tokenizer, quantization_method = "q4_k_m")