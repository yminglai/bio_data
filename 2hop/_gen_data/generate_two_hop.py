import json
import argparse
import os
from pathlib import Path

RELATION_TEMPLATES = {
    'friend_of': [
        '{A} and {B} are best pals.',
        'Everyone knows that {A} is a close friend of {B}.',
        '{A} hangs out a lot with {B}.',
        '{A} confides in {B} about personal matters.',
        '{A} has maintained a strong friendship with {B}.',
        'People often see {A} and {B} spending weekends together.',
        '{A} has known {B} since childhood and remains very close.',
        '{A} trusts {B} deeply as a friend.'
    ],
    'works_with': [
        '{A} and {B} collaborate on work projects.',
        '{A} often shares tasks with {B} in the office.',
        '{A} consults with {B} about ongoing assignments.',
        '{A} and {B} are on the same team at the company.',
        '{A} discusses business strategies regularly with {B}.',
        '{A} partners with {B} to meet deadlines.',
        '{A} has daily stand-up meetings with {B}.',
        '{A} relies on {B} for professional support at work.'
    ],
    'neighbor_of': [
        '{A} lives on the same street as {B}.',
        '{A} often sees {B} when walking the dog.',
        '{A} and {B} share a fence line.',
        '{A} occasionally borrows tools from {B}.',
        '{A} and {B} greet each other in the hallway of their apartment building.',
        '{A} frequently bumps into {B} in the lobby.',
        '{A} can see {B}\'s house from the living room window.',
        '{A} and {B} sometimes chat over the backyard fence.'
    ],
    'cousin_of': [
        '{A} is a cousin of {B}.',
        '{A} and {B} have the same grandmother.',
        '{A} attends large family gatherings with {B}.',
        '{A} shares part of the family tree with {B}.',
        '{A} grew up visiting {B}\'s home on holidays.',
        '{A} and {B} are related through their fathers.',
        '{A} spent summers with {B} at their grandparents\' farm.',
        '{A} is part of the same extended family as {B}.'
    ],
    'classmate_of': [
        '{A} and {B} have classes together.',
        '{A} takes notes alongside {B} in lectures.',
        '{A} has worked on group projects with {B} at school.',
        '{A} and {B} are on the same class roster.',
        '{A} occasionally copies homework solutions from {B}.',
        '{A} sees {B} every day in homeroom.',
        '{A} and {B} study in the same university course.',
        '{A} meets {B} frequently in the campus library.'
    ],
    'boss_of': [
        '{A} supervises {B} at work.',
        '{A} holds performance reviews for {B}.',
        '{A} controls the schedule of {B}.',
        '{A} delegates tasks to {B} on a daily basis.',
        '{A} is responsible for the promotions of {B}.',
        '{A} signs off on the expenses filed by {B}.',
        '{A} manages the team that includes {B}.',
        '{A} decides {B}\'s annual bonus.'
    ],
    'mentor_of': [
        '{A} guides {B}\'s career path.',
        '{A} offers regular counsel to {B}.',
        '{A} teaches important skills to {B}.',
        '{A} provides feedback to help {B} improve.',
        '{A} shares professional connections with {B}.',
        '{A} helps {B} grow in their role.',
        '{A} schedules regular mentorship sessions with {B}.',
        '{A} invests time in shaping {B}\'s development.'
    ],
    'admires': [
        '{A} admires {B}\'s expertise.',
        '{A} looks up to {B}\'s accomplishments.',
        '{A} is inspired by {B}\'s dedication.',
        '{A} respects {B} for their achievements.',
        '{A} praises the hard work of {B}.',
        '{A} wants to follow in {B}\'s footsteps.',
        '{A} finds {B}\'s success motivating.',
        '{A} is impressed by {B}\'s talents.'
    ],
    'has_crush_on': [
        '{A} secretly likes {B} a lot.',
        '{A} can\'t stop blushing around {B}.',
        '{A} daydreams about being with {B}.',
        '{A} feels butterflies whenever {B} appears.',
        '{A} hopes to ask {B} out someday.',
        '{A} constantly thinks of spending time with {B}.',
        '{A} is smitten with {B}.',
        '{A} shares romantic feelings for {B}.'
    ],
    'competes_with': [
        '{A} competes with {B} in tournaments.',
        '{A} and {B} are rivals in many events.',
        '{A} tries to surpass {B}\'s performance.',
        '{A} frequently challenges {B} to do better.',
        '{A} and {B} compete for the top position.',
        '{A} is determined to defeat {B}.',
        '{A} trains to outperform {B}.',
        '{A} and {B} push each other in competitions.'
    ], 
    'reports_to': [
        '{A} needs to send regular updates to {B}.',
        '{A} works under the supervision of {B}.',
        '{A} is accountable to {B} on all tasks.',
        '{A} attends one-on-one meetings with {B} to discuss progress.',
        '{A} escalates issues to {B} when problems arise.',
        '{A} consults {B} before making major decisions.',
        '{A} files weekly performance reports with {B}.',
        '{A} has {B} as the main decision-maker on the team.'
    ],
    'follows': [
        '{A} subscribes to updates from {B} on social media.',
        '{A} keeps track of content posted by {B}.',
        '{A} is influenced by the online presence of {B}.',
        '{A} checks notifications whenever {B} posts something new.',
        '{A} regularly reads the blog articles of {B}.',
        '{A} gains inspiration by following the work of {B}.',
        '{A} encourages others to also follow {B}.',
        '{A} learns from the tutorials or content that {B} shares.'
    ],
    'owes_debt_to': [
        '{A} must repay an outstanding amount to {B}.',
        '{A} borrowed funds from {B} in the past.',
        '{A} is concerned about the interest charged by {B}.',
        '{A} signed a loan agreement with {B}.',
        '{A} made a partial payment to {B} but still owes more.',
        '{A} has an overdue balance with {B}.',
        '{A} is on a repayment schedule set by {B}.',
        '{A} acknowledged the debt owed to {B}.'
    ],
    'subscribes_to': [
        '{A} signed up for the mailing list of {B}.',
        '{A} receives newsletters curated by {B}.',
        '{A} pays a monthly subscription fee to {B}.',
        '{A} looks forward to updates generated by {B}.',
        '{A} shares subscription perks from {B} with friends.',
        '{A} has an automatic renewal with the service of {B}.',
        '{A} finds valuable content through {B}’s channel.',
        '{A} leaves feedback to help {B} improve offerings.'
    ],
    'endorsed_by': [
        '{A} lists {B} as an official sponsor.',
        '{A} uses the testimonials provided by {B}.',
        '{A} proudly displays the logo of {B} on social media.',
        '{A} speaks highly of the support given by {B}.',
        '{A} credits {B} for validating their skills or product.',
        '{A} appreciates the public backing offered by {B}.',
        '{A} hopes to maintain a positive relationship with {B}.',
        '{A} counts on {B}’s endorsement for credibility.'
    ],
    'blames': [
        '{A} holds {B} responsible for the failure.',
        '{A} complains that {B} caused the issue.',
        '{A} believes {B} made a critical mistake.',
        '{A} points fingers at {B} whenever something goes wrong.',
        '{A} accuses {B} of negligence.',
        '{A} publicly states that {B} messed up.',
        '{A} insists the fault lies with {B}.',
        '{A} focuses on the errors committed by {B}.'
    ],
    'accuses': [
        '{A} suspects {B} of wrongdoing.',
        '{A} filed a formal complaint against {B}.',
        '{A} believes {B} may have violated a rule.',
        '{A} publicly questions the motives of {B}.',
        '{A} brings charges or allegations against {B}.',
        '{A} demands an explanation from {B} about the incident.',
        '{A} claims that {B} acted dishonestly.',
        '{A} calls for an investigation into {B}’s actions.'
    ],
    'forgives': [
        '{A} accepts an apology from {B}.',
        '{A} decides to move on from the conflict with {B}.',
        '{A} offers a second chance to {B}.',
        '{A} no longer resents {B} for the mistake.',
        '{A} believes that {B} deserves redemption.',
        '{A} puts the past behind and reconciles with {B}.',
        '{A} embraces the peace accord with {B}.',
        '{A} acknowledges that {B} has changed for the better.'
    ],
    'warns': [
        '{A} cautions {B} about potential dangers.',
        '{A} alerts {B} to stay away from suspicious areas.',
        '{A} issues a safety notice to {B}.',
        '{A} urges {B} to be careful when traveling.',
        '{A} advises {B} to reconsider a risky decision.',
        '{A} shares urgent news to protect {B}.',
        '{A} sends a heads-up to {B} via text.',
        '{A} sounds the alarm to make {B} aware of threats.'
    ],
    'protects': [
        '{A} stands guard over {B} in dangerous situations.',
        '{A} shields {B} from any external threats.',
        '{A} commits to keeping {B} safe at all costs.',
        '{A} uses personal resources to defend {B}.',
        '{A} steps in whenever {B} is in jeopardy.',
        '{A} prioritizes the well-being of {B}.',
        '{A} sacrifices time and effort to safeguard {B}.',
        '{A} refuses to let any harm come to {B}.'
    ],
}

def load_train_qa_data(path):
    """
    read train_qa_data.json,
    return a list
    format:
        "question": question,
        "answer": answer,
        "fact": fact_list
    example:
        "question": "Who is the friend of the report of Avery?",
        "answer": "Gerald",
        "fact": [
            "Avery", // Entity_1
            "reports_to", // Relation_1
            "Dominic",  // Entity_2
            "friend_of", // Relation_2
            "Gerald"    // Entity_3
        ]
    """
    
    with open(path, "r") as f:
        raw = json.load(f)

    dataset = []

    for item in raw:
        question = item.get("question")
        answer = item.get("answer")
        fact_list = item.get("fact", [])
        dataset.append({
            "question": question,
            "answer": answer,
            "fact": fact_list
        })

    return dataset


def replace_relations_by_position(sentence: str):
    """
    Replace relation words based on fixed positions in the sentence.
    
    In a template sentence with the structure:
        "Who is the X of the Y of [ENTITY_1]?"
    
    - The 4th token (index 3) corresponds to X → replaced with [RELATION_2]
    - The 7th token (index 6) corresponds to Y → replaced with [RELATION_1]
    
    This function does NOT rely on matching specific words such as
    'subscribe' or 'mentor'. It only uses positional replacement.
    """
    tokens = sentence.split()

    # Example tokenization:
    # ["Who", "is", "the", "subscribe", "of", "the", "mentor", "of", "[ENTITY_1]?"]
    # Index mapping:
    #    0      1      2       3          4      5      6        7        8
    relation_2_original = tokens[3]  # Store original relation at position 3
    relation_1_original = tokens[6]  # Store original relation at position 6
    tokens[3] = "[RELATION_2]"   # Replace the relation placeholder at position 3
    tokens[6] = "[RELATION_1]"   # Replace the relation placeholder at position 6
    tokens[8] = "[ENTITY_1]"

    return " ".join(tokens), relation_1_original, relation_2_original

def process_data(dataset):
    """
    Original data format example:
        "question": "Who is the friend of the report of Avery?",
        "answer": "Gerald",
        "fact": [
            "Avery", // Entity_1
            "reports_to", // Relation_1
            "Dominic",  // Entity_2
            "friend_of", // Relation_2
            "Gerald"    // Entity_3
        ]
    Change the data format into :
        1. Question template
        2. Entity_1
        3. Relation_1
        4. Relation_2
        5. Entity_2
        6. Entity_3
        7. Question (fill question template with Entity_1, Relation_1, Relation_2)
        8. Output (Entity_2, Entity_3)
        9. relation_1_sentences
        10. relation_2_sentences
    """
    new_data = []
    for item in dataset:
        # step0: prepare variables
        question = item["question"]         # 7. Question
        answer = item["answer"]
        entity_1 = item["fact"][0]          # 2. Entity_1
        relation_1 = item["fact"][1]        # 3. Relation_1
        entity_2 = item["fact"][2]          # 5. Entity_2
        relation_2 = item["fact"][3]        # 4. Relation_2
        entity_3 = item["fact"][4]          # 6. Entity_3
        # step1: remove the Entity_1 from the question to create Question template
        question_template, relation_1_original, relation_2_original = replace_relations_by_position(question) # 1. Question template
        # step2: generate output
        output = f"{entity_2} {entity_3}"       # 8. Output: "Entity_2 Entity_3"

        # step3: use relation_1 and RELAION_TEMPLATES to generate relation_1_sentence
        relation_1_sentences_templates = RELATION_TEMPLATES[relation_1]
        relation_1_sentences=[] # 9. relation_1_sentences
        for template in relation_1_sentences_templates:
            relation_1_sentence = template.format(A=entity_2, B=entity_1)  
            relation_1_sentences.append(relation_1_sentence)
        # step4: use relation_2 and RELAION_TEMPLATES to generate relation_2_sentence
        relation_2_sentences_templates = RELATION_TEMPLATES[relation_2]
        relation_2_sentences=[]  # 10. relation_2_sentences
        for template in relation_2_sentences_templates:
            relation_2_sentence = template.format(A=entity_3, B=entity_2)
            relation_2_sentences.append(relation_2_sentence)
        
        
        # step4: form the new data format
        new_data_format = {
            "question_template": question_template,
            "entity_1": entity_1,
            "relation_1": relation_1,
            "relation_2": relation_2,
            "entity_2": entity_2,
            "entity_3": entity_3,
            "question": question,
            "output": output,
            "relation_1_sentences": relation_1_sentences,
            "relation_2_sentences": relation_2_sentences,
            "original_relations": {
                "relation_1_original": relation_1_original,
                "relation_2_original": relation_2_original
            }
        }
        new_data.append(new_data_format)
    return new_data


def save_to_jsonl(data, path):
    """
    Save the processed data to a JSONL file.
    Each line in the file corresponds to one JSON object.
    """
    with open(path, "w") as f:
        for item in data:
            json_line = json.dumps(item)
            f.write(json_line + "\n")
    print(f"Data saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to input QA json file (must contain train or val)"
    )
    args = parser.parse_args()

    # Output goes to 2hop/_dataset/_gen/ regardless of the caller's cwd.
    out_dir = Path(__file__).resolve().parent.parent / "_dataset" / "_gen"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = os.path.basename(args.path)
    if "train" in filename:
        save_path = str(out_dir / "train_two_hop_qa_data.jsonl")
    elif "val" in filename:
        save_path = str(out_dir / "val_two_hop_qa_data.jsonl")
    else:
        raise ValueError("Input filename must contain 'train' or 'val' to determine save path.")

        
    # gen
    dataset = load_train_qa_data(args.path)
    new_format_dataset = process_data(dataset)
    save_to_jsonl(new_format_dataset, save_path)


    