import pandas as pd
from datasets import Dataset, DatasetDict
import os

def process_symptoms(row):
    """Convert symptom columns into a list of non-empty symptoms."""
    symptoms = []
    for col in row.index[1:]:  # Skip the Disease column
        symptom = str(row[col]).strip()
        if symptom and symptom != 'nan' and symptom != '':
            symptoms.append(symptom)
    return symptoms

def main():
    # File paths
    input_file = 'DiseaseAndSymptoms.csv'
    output_dir = 'disease_symptoms_dataset'
    
    # Read the CSV file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Process the data
    print("Processing data...")
    # Get all unique diseases
    diseases = df['Disease'].unique()
    
    # Create lists to store the data
    disease_list = []
    symptoms_list = []
    
    # Process each row
    for _, row in df.iterrows():
        disease = row['Disease'].strip()
        symptoms = process_symptoms(row)
        
        if symptoms:  # Only add if there are symptoms
            disease_list.append(disease)
            symptoms_list.append(symptoms)
    
    # Create dataset
    dataset_dict = {
        'disease': disease_list,
        'symptoms': symptoms_list
    }
    
    # Convert to Hugging Face dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train and test (80-20 split)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset locally
    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)
    
    print("\nDataset structure:")
    print(dataset)
    
    print("\nExample entries:")
    for i in range(3):
        print(f"Disease: {dataset['train']['disease'][i]}")
        print(f"Symptoms: {', '.join(dataset['train']['symptoms'][i])}\n")
    
    # Check if we should push to the Hub
    push_to_hub = input("\nDo you want to push this dataset to the Hugging Face Hub? (y/n): ").lower()
    if push_to_hub == 'y':
        from huggingface_hub import HfApi, HfFolder
        
        if not HfFolder.get_token():
            print("\nPlease log in to Hugging Face Hub first.")
            print("Run: huggingface-cli login")
            print("Or in Python: from huggingface_hub import notebook_login; notebook_login()")
            return
            
        repo_name = input("Enter the name for your dataset on the Hub (e.g., username/disease-symptoms): ")
        
        print(f"Pushing dataset to {repo_name}...")
        dataset.push_to_hub(repo_name)
        print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_name}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
