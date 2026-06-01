from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    
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
        ex["text"] = f"I have {csv_symptoms}. I might be experiencing {disease}"
        # ex["text"] = f"The symptoms of {disease} are {csv_symptoms}"
        # ex["text"] = f"Disease: {disease}\nSymptoms: {csv_symptoms}\n"
        return ex
    
    dataset = dataset.map(initialize)
    
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = dataset.map(tokenize, batched=True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir="./third_fine_tuned_gpt2",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1, # Key argument for performance
        save_steps=500,
        save_total_limit=2,
        logging_steps=100
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        processing_class=tokenizer
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
