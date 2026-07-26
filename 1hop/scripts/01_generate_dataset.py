"""
Step 1: Generate Knowledge Graph with Train/Test Split
Creates structured dataset with 1 template for training, 3 for testing per rule.
"""
import csv
import random
import os
import json
import datetime
from pathlib import Path

def load_list(filename):
    """Loads a list from a file, stripping whitespace and trailing periods."""
    with open(filename, "r") as f:
        return [line.strip().rstrip('.') for line in f if line.strip()]

def get_random_date():
    """Generates a random date and returns it in Day,Month,Year format."""
    year = random.randint(1950, 2005)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    date_obj = datetime.datetime(year, month, day)
    return f"{date_obj.day},{date_obj.strftime('%B')},{date_obj.year}"

def generate_biography(person_data, templates):
    """Generates a formatted biography for a person."""
    pronoun_map = templates["pronouns"][person_data["pronoun_key"]]
    capitalized_name = " ".join(word.capitalize() for word in person_data["full_name"].split())

    first_sentence_template_data = {
        "PRONOUN": capitalized_name,
        "PRONOUN_POSSESSIVE": f"{capitalized_name}'s",
        "BIRTH_DATE": person_data["birth_date"],
        "FULL_NAME": capitalized_name,
    }
    first_sentence = random.choice(templates["birth_date"]).format(**first_sentence_template_data)

    template_data = {
        "FULL_NAME": capitalized_name,
        "BIRTH_DATE": person_data["birth_date"],
        "BIRTH_CITY": person_data["birth_city"],
        "UNIVERSITY": person_data["university"],
        "MAJOR": person_data["major"],
        "EMPLOYER": person_data["employer"],
        "COMPANY_CITY": person_data["work_city"],
        **pronoun_map
    }

    sentence_templates = {
        "birth_city": templates["birth_city"],
        "university": templates["university"],
        "major": templates["major"],
        "employer": templates["employer"],
        "company_city": templates["company_city"],
    }
    
    sentences = [random.choice(sentence_templates[key]).format(**template_data) for key in sentence_templates]
    random_attribute_templates = random.choice(list(sentence_templates.values()))
    sentences.append(random.choice(random_attribute_templates).format(**template_data))
    random.shuffle(sentences)

    all_sentences = [first_sentence] + sentences
    return ". ".join(all_sentences) + "."

def main():
    random.seed(42)
    
    # Configuration
    num_persons = 1000  # Total persons to generate
    train_ratio = 0.5   # 50% train, 50% test
    
    data_dir = Path("data/entities")
    templates_dir = Path("data/templates")
    qa_templates_dir = Path("data/qa_templates")
    output_dir = Path("data/generated")
    output_dir.mkdir(exist_ok=True)

    # Load entities
    first_names = load_list(data_dir / "first_names.txt")
    middle_names = load_list(data_dir / "middle_names.txt")
    last_names = load_list(data_dir / "last_names.txt")
    birth_cities = load_list(data_dir / "birth_cities.txt")
    universities = load_list(data_dir / "universities.txt")
    major_names = load_list(data_dir / "major_names.txt")

    companies = []
    with open(data_dir / "companies.csv", "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                companies.append({"name": row[0], "city": row[1]})

    # Load biography templates
    templates = {
        "birth_date": load_list(templates_dir / "birth_date_templates.txt"),
        "birth_city": load_list(templates_dir / "birth_city_templates.txt"),
        "university": load_list(templates_dir / "university_templates.txt"),
        "major": load_list(templates_dir / "major_templates.txt"),
        "employer": load_list(templates_dir / "employer_templates.txt"),
        "company_city": load_list(templates_dir / "company_city_templates.txt"),
        "pronouns": {
            "he": {"PRONOUN": "He", "PRONOUN_POSSESSIVE": "his"},
            "she": {"PRONOUN": "She", "PRONOUN_POSSESSIVE": "her"}
        }
    }

    # Load Q&A templates (we'll split these into train/test)
    qa_templates = {
        "birth_date": load_list(qa_templates_dir / "birth_date_questions.txt"),
        "birth_city": load_list(qa_templates_dir / "birth_city_questions.txt"),
        "university": load_list(qa_templates_dir / "university_questions.txt"),
        "major": load_list(qa_templates_dir / "major_questions.txt"),
        "employer": load_list(qa_templates_dir / "employer_questions.txt"),
        "company_city": load_list(qa_templates_dir / "company_city_questions.txt"),
    }

    # Generate unique persons
    all_persons = []
    used_names = set()
    
    while len(all_persons) < num_persons:
        first = random.choice(first_names).capitalize()
        middle = random.choice(middle_names)
        last = random.choice(last_names)
        full_name = f"{first} {middle} {last}".title()

        if full_name in used_names:
            continue
        used_names.add(full_name)

        company_info = random.choice(companies)
        pronoun_key = random.choice(list(templates["pronouns"].keys()))

        person = {
            "person_id": f"person_{len(all_persons):04d}",
            "full_name": full_name,
            "birth_date": get_random_date(),
            "birth_city": random.choice(birth_cities),
            "university": random.choice(universities),
            "major": random.choice(major_names),
            "employer": company_info["name"],
            "work_city": company_info["city"],
            "pronoun_key": pronoun_key
        }
        
        # Generate 5 biography variants
        person["biographies"] = [
            generate_biography(person, templates) 
            for _ in range(5)
        ]
        
        all_persons.append(person)

    # Save knowledge graphs (person records incl. biographies).
    # 02 (SFT) reads train_kg.json for biography memorization;
    # 05/06 read test_kg.json to look up swap-target attributes.
    # Train and test share the same persons - the train/test split
    # is over QA templates (0-1 train, 2-3 held-out), not persons.
    with open(output_dir / "train_kg.json", "w") as f:
        json.dump(all_persons, f, indent=2)
    with open(output_dir / "test_kg.json", "w") as f:
        json.dump(all_persons, f, indent=2)
    print(f"Saved knowledge graphs for {len(all_persons)} persons (train_kg.json, test_kg.json)")

    # Generate QA pairs for ALL persons with proper template splits
    # Train: templates 0,1 for all persons (to train SAE on ID templates)
    # Test-OOD: templates 2,3 for all persons (different phrasings)
    
    rule_names = ["birth_date", "birth_city", "university", "major", "employer", "company_city"]
    rule_to_person_key = {
        "birth_date": "birth_date",
        "birth_city": "birth_city",
        "university": "university",
        "major": "major",
        "employer": "employer",
        "company_city": "work_city"
    }
    
    qa_train = []  # Templates 0,1 for all persons
    qa_test_ood = []  # Templates 2,3 for all persons

    for person in all_persons:
        for rule_idx, rule_name in enumerate(rule_names):
            templates_for_rule = qa_templates[rule_name]
            
            # Templates 0,1: Training (ID templates)
            for template_idx in [0, 1]:
                if template_idx < len(templates_for_rule):
                    question = templates_for_rule[template_idx].format(FULL_NAME=person["full_name"])
                    person_key = rule_to_person_key[rule_name]
                    answer = person[person_key]
                    qa_train.append({
                        "person_id": person["person_id"],
                        "full_name": person["full_name"],
                        "rule_idx": rule_idx,
                        "rule_name": rule_name,
                        "question": question,
                        "answer": answer,
                        "template_idx": template_idx,
                        "split": "train",
                        "is_ood": False
                    })
            
            # Templates 2,3: Test-OOD (different phrasings)
            for template_idx in [2, 3]:
                if template_idx < len(templates_for_rule):
                    question = templates_for_rule[template_idx].format(FULL_NAME=person["full_name"])
                    person_key = rule_to_person_key[rule_name]
                    answer = person[person_key]
                    qa_test_ood.append({
                        "person_id": person["person_id"],
                        "full_name": person["full_name"],
                        "rule_idx": rule_idx,
                        "rule_name": rule_name,
                        "question": question,
                        "answer": answer,
                        "template_idx": template_idx,
                        "split": "test_ood",
                        "is_ood": True
                    })

    # Save QA datasets
    with open(output_dir / "qa_train.jsonl", "w") as f:
        for qa in qa_train:
            f.write(json.dumps(qa) + "\n")
    with open(output_dir / "qa_test_ood.jsonl", "w") as f:
        for qa in qa_test_ood:
            f.write(json.dumps(qa) + "\n")

    print(f"\nGenerated QA pairs:")
    print(f"  Train (templates 0-1): {len(qa_train)} (2 templates × 6 rules × {len(all_persons)} persons)")
    print(f"  Test-OOD (templates 2-3): {len(qa_test_ood)} (2 templates × 6 rules × {len(all_persons)} persons)")
    print(f"  Total: {len(qa_train) + len(qa_test_ood)}")
    
    # Save combined biography file for all persons
    with open(output_dir / "biographies_all.txt", "w") as f:
        for person in all_persons:
            for bio in person["biographies"]:
                f.write(f"{bio}\n")

    print(f"\nDataset generation complete! Files saved to {output_dir}/")

if __name__ == "__main__":
    main()
