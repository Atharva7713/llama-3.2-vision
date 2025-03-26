import os
import json
import torch
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from huggingface_hub import login

login("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") # this is the access token from hugging to access the llama model 

# Disable SDPA Flash Attention to avoid unpacking error (you can remove if you have high gpu)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)  # Add this line to fully disable flash attention

MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision"
OUTPUT_DIR = "./llama-3.2-fine-tuned-new"        # if you want to store the fine tuned folder in another folder then simply give the path of that folder where you want to store the fine tuned model with forward slash.
DATASET_PATH = "./updated_train_data (1).jsonl"   # to change the location simply remove the previous file path and write the new path but everything should only have forward slash in complete path.

# example: if path is this C:\Users\user\Desktop\data_preperation\llama-3.2-finetune-model.jsonl change it like this C:/Users/user/Desktop/data_preperation/llama-3.2-finetune-model.jsonl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Load model with proper initialization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=False,
    bnb_4bit_compute_dtype=torch.float16
)

config = AutoConfig.from_pretrained(MODEL_NAME)  # Ensure it's a config object

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    #use_cache=False,  # Ensure compatibility with gradient checkpointing
    attn_implementation="eager"  # Use eager implementation instead of SDPA
)

# Ensure model is properly initialized before moving to CUDA
model = model.to(device)
model.resize_token_embeddings(len(tokenizer))

# Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def preprocess_function(example):
    messages = example["messages"]
    conversation = ""
    
    # Format the conversation with proper structure and clear role separation
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role.lower() == "user":
            conversation += f"### User: {content}\n"
        elif role.lower() == "assistant":
            conversation += f"### Assistant: {content}\n"
    
    # Add EOS token at the end
    conversation = conversation.strip() + tokenizer.eos_token
    
    # Tokenize with appropriate length handling
    tokenized_inputs = tokenizer(
        conversation,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Create proper labels
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return {
        "input_ids": tokenized_inputs["input_ids"][0],
        "attention_mask": tokenized_inputs["attention_mask"][0],
        "labels": tokenized_inputs["labels"][0]
    }

dataset = load_jsonl(DATASET_PATH)
train_tokenized = [preprocess_function(example) for example in dataset]

from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {key: val.clone().detach() for key, val in self.data[idx].items()}

train_dataset = CustomDataset(train_tokenized)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=5,           # change the epochs here if required
    per_device_train_batch_size=2,  # Reduced from 4 to help with memory
    gradient_accumulation_steps=8,
    save_steps=500,
    save_total_limit=3,
    fp16=True,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Explicitly set use_reentrant
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# Custom stopping criteria to ensure model stops properly at role transitions
class UserStartStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_token_ids:
            if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                return True
        return False

def generate_text(input_text, max_length=200):
    # Ensure input has proper formatting
    if not input_text.startswith("### User:"):
        input_text = f"### User: {input_text}"
    
    # Ensure the input ends with assistant prompt
    if "### Assistant:" not in input_text:
        input_text = f"{input_text.strip()}\n### Assistant:"
    
    model.eval()  # Set to evaluation mode
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Create stopping criteria to avoid generating a new user turn
    user_turn_ids = tokenizer.encode("\n### User:", add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList([
        UserStartStoppingCriteria([[id] for id in user_turn_ids])
    ])
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the assistant's response
    try:
        assistant_response = full_response.split("### Assistant:")[-1].split("\n### User:")[0].strip()
    except IndexError:
        assistant_response = full_response.split("### Assistant:")[-1].strip()
    
    return assistant_response

# Test the model
if __name__ == "__main__":
    prompt = "### User: Explain clinical trials.\n### Assistant:"
    generated = generate_text(prompt)
    print("Input prompt:")
    print(prompt)
    print("\nGenerated response:")
    print(generated)
    
    # Test with another example
    prompt2 = "### User: What are the benefits of regular exercise?\n### Assistant:"
    generated2 = generate_text(prompt2)
    print("\nInput prompt 2:")
    print(prompt2)
    print("\nGenerated response 2:")
    print(generated2)
