import json
import os
import csv
import random
import openai
import sys
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Function to use GPT to generate Q&A pairs
def ask_gpt(prompt, model="gpt-4o-mini", temperature=0.3):
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in generating question-answer pairs based on text."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Function to generate QA pairs using GPT with token limit and sentence-based count (For the CLIP-based model)
def generate_qa_pairs(description, num_questions=3):
    # Count sentences in the description to determine question count
    sentences = sent_tokenize(description)
    sentence_count = len(sentences)
    # Use sentence count if provided, otherwise use default
    question_count = sentence_count if sentence_count > 0 else num_questions
    
    prompt = f"""
    Based on the following description of a traffic scenario, generate {question_count} question-answer pairs.
    Make the questions specific, informative, and test understanding of the scenario.
    The answers should be detailed and accurate. Each question and each answer must be under 77 tokens.
    Below is an image caption. Your task is to automatically generate high-quality question–answer pairs following the VQ2A methodology. Please adhere strictly to the following three steps:

    Candidate Answer Extraction:

    Parse the caption and extract candidate answer spans. Include:
        Noun phrases and named entities.
        Sequences (POS spans) that start and end with open-class words (nouns, verbs, adjectives, adverbs), allowing only determiners, adpositions, or conjunctions in between.
        Short parse tree spans (maximal subtrees of up to 3 words that contain at least one open-class word).
        Boolean candidates “yes” and “no” (even if they do not appear in the caption).
        For “how many” questions, if a zero count is not evident, consider introducing “zero” by sampling from a similar caption context.

    Question Generation:
        For each candidate answer, generate a question by transforming the caption’s declarative sentence into an interrogative form such that the candidate answer is the correct answer when the caption is the context.
        Ensure that questions cover various types (e.g., “How many…”, “What is…”, “Where are…?”, “Is there…?”) as appropriate for the information in the caption.
    
    Answer Validation (Round-Trip Check):
        Simulate answering each generated question using the caption as context.
        Only retain a question–answer pair if the answer obtained from the caption (via a simulated question-answering process) matches the candidate answer with a token-level F1 score above 0.54.
        Discard any pairs that do not pass this round-trip validation.

    For example, given the caption:
    "Two bears are laying down on the ice."
    A proper output might include pairs such as:
    • Q: "How many bears are laying on the ice?" A: "two"
    • Q: "What are the two animals on the ice?" A: "bears"
    • Q: "What are the bears doing?" A: "laying down"
    • Q: "Where are the bears laying?" A: "on the ice"

    Now, please process the following caption and output all validated question–answer pairs:
    
    Description: {description}
    
    Return your response in the following JSON format:
    {{
        "qa_pairs": [
            {{"question": "question 1", "answer": "answer 1"}},
            {{"question": "question 2", "answer": "answer 2"}},
            ...
        ]
    }}
    """
    
    try:
        result = ask_gpt(prompt)
        qa_pairs = result.get("qa_pairs", [])
        print(f"Generated {len(qa_pairs)} QA pairs")
        return qa_pairs
    except Exception as e:
        print(f"Error generating QA pairs: {e}")
        return []

# Load video descriptions from a JSON file per video
def load_video_descriptions(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    video_names = []
    descriptions = []
    for item in data:
        video_names.append(item['video_name'])
        descriptions.append(item['response'])
    
    print(f"Loaded {len(video_names)} videos and descriptions from {file_name}")
    return video_names, descriptions

# Load original SUTD data with vid_id and perspective from a jsonl file that contains lists.
def load_sutd_original_data(sutd_ori_file):
    vid_info = {}
    
    # Check if file exists
    if not os.path.exists(sutd_ori_file):
        print(f"Warning: File {sutd_ori_file} does not exist.")
        return vid_info
    
    with open(sutd_ori_file, 'r', encoding='utf-8') as f:
        # Process jsonl file line by line
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                
                # Check if the item is a list
                if isinstance(item, list):
                    if len(item) >= 4: 
                        vid_id = item[1]
                        vid_filename = item[2]
                        perspective = item[3]
                        
                        if vid_filename:
                            vid_info[vid_filename] = {
                                'vid_id': vid_id,
                                'perspective': perspective
                            }
                else:
                    # In case some lines are dictionaries
                    vid_filename = item.get('vid_filename')
                    if vid_filename:
                        vid_info[vid_filename] = {
                            'vid_id': item.get('vid_id'),
                            'perspective': item.get('perspective', 1)  # Default to 1 if missing
                        }
            except json.JSONDecodeError:
                print(f"Error parsing line {line_num} in {sutd_ori_file}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num} in {sutd_ori_file}: {e}")
                continue
    
    print(f"Loaded information for {len(vid_info)} videos from {sutd_ori_file}")
    return vid_info

# Process descriptions and create multiple-choice questions with options.
def process_and_save_final_format(video_names, descriptions, vid_info, output_file):
    # First, collect all question-answer pairs from all videos
    all_qa_pairs_by_video = {}
    all_answers = []
    k = 0
    print("First pass: collecting all Q&A pairs...")
    for i, description in enumerate(descriptions):
        vid_filename = video_names[i]
        try:
            # Process the description with GPT instead of NLP pipeline
            qna_result = generate_qa_pairs(description)
            all_qa_pairs_by_video[vid_filename] = qna_result
            
            # Extract just the answers for sampling options later
            for qa_pair in qna_result:
                all_answers.append(qa_pair["answer"])
        except Exception as e:
            print(f"Error collecting QA pairs for {vid_filename}: {e}")
            all_qa_pairs_by_video[vid_filename] = []
            k+=1
    
    # Now generate the final data with multiple-choice options
    final_data = []
    record_id = 0
    
    print("Second pass: generating multiple-choice questions...")
    for i, description in enumerate(descriptions):
        vid_filename = video_names[i]
        video_data = vid_info.get(vid_filename, {'vid_id': 0, 'perspective': 1})
        
        # Get the QA pairs for this video
        qna_result = all_qa_pairs_by_video[vid_filename]
        
        # Process each question-answer pair
        for qa_pair in qna_result:
            question = qa_pair["question"]
            correct_answer = qa_pair["answer"]
            
            # Sample 3 random answers from other videos (excluding this answer)
            other_options = []
            potential_options = [ans for ans in all_answers if ans != correct_answer]
            
            # If we don't have enough options, create some variations
            if len(potential_options) < 3:
                for _ in range(3 - len(potential_options)):
                    modified = f"Not {correct_answer}" if not correct_answer.startswith("Not ") else correct_answer[4:]
                    potential_options.append(modified)
            
            # Sample 3 unique options
            sampled_options = []
            for _ in range(3):
                if potential_options:
                    option = random.choice(potential_options)
                    sampled_options.append(option)
                    # Remove to avoid duplicates
                    potential_options.remove(option)
            
            # Combine correct answer with sampled options
            all_options = [correct_answer] + sampled_options
            
            # Shuffle the options
            random.shuffle(all_options)
            
            # Find the index of the correct answer after shuffling
            correct_index = all_options.index(correct_answer)
            
            # Create the record
            record = [
                record_id,  # record_id
                video_data['vid_id'],  # vid_id
                vid_filename,  # vid_filename
                video_data['perspective'],  # perspective
                question,  # q_body
                "A",  # q_type (always A as specified)
                all_options[0],  # option0
                all_options[1],  # option1
                all_options[2],  # option2
                all_options[3],  # option3
                correct_index  # answer index (position of correct answer)
            ]
            
            final_data.append(record)
            record_id += 1
    
    # Define the header row
    header = ["record_id", "vid_id", "vid_filename", "perspective", "q_body", "q_type", 
              "option0", "option1", "option2", "option3", "answer"]
    
    # Save to JSONL file with header as first line
    jsonl_file = output_file.replace('.json', '.jsonl')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        # Write header as first line
        f.write(json.dumps(header) + '\n')
        
        # Write data rows
        for record in final_data:
            f.write(json.dumps(record) + '\n')
    
    # Also save as CSV for easier viewing
    csv_file = output_file.replace('.json', '.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(final_data)
    
    print(f"Saved {len(final_data)} multiple-choice questions to {jsonl_file} and {csv_file}")
    print("Excluded videos: ", k)
    return final_data

if __name__ == "__main__":
    # Set your OpenAI API key in environment variable
    os.environ["OPENAI_API_KEY"] = "please set your own openai api key here"
    
    input_file = "blank_removed_SUTD.json"
    sutd_ori_file = "data/sutd-traffic/output_file_train.jsonl"
    output_file = "final_SUTD_qa_without_blank_gpt_under_token77_allyoumayneed.json"

    # Step 1: Load the video descriptions
    video_names, descriptions = load_video_descriptions(input_file)

    # Step 2: Load the original SUTD data
    vid_info = load_sutd_original_data(sutd_ori_file)

    # Step 3: Process and save in the final format
    results = process_and_save_final_format(video_names, descriptions, vid_info, output_file)