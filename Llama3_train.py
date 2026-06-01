from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

def main():
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("aaditya/Llama3-OpenBioLLM-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb = BitsAndBytesConfig(load_in_4bit=True) # 4 bit weight quantization for efficiency
    
    model = AutoModelForCausalLM.from_pretrained(
        "aaditya/Llama3-OpenBioLLM-8B",
        quantization_config=bnb, # 4 bit weight quantization for efficiency
        device_map="auto"
    )
    
    model = get_peft_model(model, lora_config)
    dataset = load_dataset("MagazinePuma/Disease_and_symptoms")
    
    def initialize(ex):
        disease = ex["disease"]
        symptoms = ex["symptoms"]
        eng_symptoms = []
    
        for symptom in symptoms:
            if "_" in symptom:
                symptom = symptom.replace("_", " ")
            eng_symptoms.append(symptom)
       
        csv_symptoms = ", ".join(eng_symptoms)
    
        # chat = [
        #     {"role": "user", "content": f"I have {csv_symptoms}. What disease might I have?"}, # Llama3 chat format to maintain instructional tuning
        #     {"role": "assistant", "content": f"You might have {disease}."}
        # ]
    
        # ex["text"] = tokenizer.apply_chat_template(chat, tokenize=False)
        ex["text"] = f"<|user|>\nI have {csv_symptoms}. What disease might I have?\n<|assistant|>\nYou might have {disease}."
        return ex
    
    dataset = dataset.map(initialize)
    
    def tokenize(batch):
        ret = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
        ret["labels"] = ret["input_ids"].copy()
    
        return ret
       
    # tokenizer.pad_token = tokenizer.eos_token
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names) # Remove columns to prevent model confusion: only plays attention to input_ids, attention_mask, and labels
    
    training_args = TrainingArguments(
        output_dir="./fine_tuned_llama8B",
        num_train_epochs=3,
        per_device_train_batch_size=1, # Key argument for performance
        learning_rate=2e-5, # Small learning rate for stable tuning
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        eval_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
