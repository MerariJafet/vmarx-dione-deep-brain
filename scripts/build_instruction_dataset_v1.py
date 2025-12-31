import json
import random
import os

# seeds for reproducibility
random.seed(42)

def gen_json_arithmetic():
    a, b = random.randint(1000, 9999), random.randint(1000, 9999)
    x, y = random.randint(10, 99), random.randint(10, 99)
    d1, d2 = random.randint(100, 999), random.randint(2, 12)
    
    instruction = f"Return a JSON with keys: add, mul, div. Compute: add={a}+{b}, mul={x}*{y}, div={d1*d2}/{d2}. Output ONLY JSON."
    output_obj = {
        "add": a + b,
        "mul": x * y,
        "div": d1
    }
    return {"instruction": instruction, "output": json.dumps(output_obj), "task_type": "json_format"}

def gen_csv_increment():
    rows = random.randint(2, 5)
    start = random.randint(1, 10)
    instruction = f"Output a CSV with header 'a,b,c' and exactly {rows} rows where a increments {start}..{start+rows-1}, b is a*10, c is a squared. Output ONLY CSV."
    lines = ["a,b,c"]
    for i in range(rows):
        a = start + i
        lines.append(f"{a},{a*10},{a*a}")
    return {"instruction": instruction, "output": "\n".join(lines), "task_type": "csv_format"}

def gen_concise_bullets():
    topics = ["gradient checkpointing", "learning rate warm-up", "weight decay", "batch normalization", "attention mechanism"]
    topic = random.choice(topics)
    instruction = f"In 5 bullet points, explain why {topic} is useful in transformer training. No more than 90 words total."
    # Synthetic placeholder response (usually you'd use a real one, but for a template script, we simulate adherence)
    output = f"- It helps with stabilization.\n- It improves convergence speed.\n- It reduces memory overhead.\n- It optimizes weight updates.\n- It prevents overfitting.\n(Total words check: useful for LLM behavior training)"
    return {"instruction": instruction, "output": output, "task_type": "concise_bullets"}

def gen_pseudocode_circuit():
    limit = random.choice([10, 20, 50, 100])
    instruction = f"Write pseudo-code for a circuit breaker that stops after {limit} consecutive NaN/Inf losses. Output ONLY pseudo-code."
    output = f"counter = 0\nfor loss in losses:\n    if is_nan(loss) or is_inf(loss):\n        counter += 1\n    else:\n        counter = 0\n    if counter >= {limit}:\n        stop_training()"
    return {"instruction": instruction, "output": output, "task_type": "pseudocode"}

def build_dataset(target_size=12000, val_size=800):
    train_data = []
    
    # Mix proportions
    # json_format: 35%, csv_format: 25%, concise_bullets: 15%, pseudocode: 10%, schema_following: 15% (omitting schema for brevity in this MVP script)
    
    generators = [
        (gen_json_arithmetic, 0.40),
        (gen_csv_increment, 0.30),
        (gen_concise_bullets, 0.15),
        (gen_pseudocode_circuit, 0.15)
    ]
    
    for _ in range(target_size + val_size):
        gen_func = random.choices([g[0] for g in generators], weights=[g[1] for g in generators])[0]
        example = gen_func()
        example["id"] = f"inst_{len(train_data)}"
        example["difficulty"] = "easy"
        example["source"] = "synthetic_template"
        train_data.append(example)
        
    random.shuffle(train_data)
    
    val_set = train_data[:val_size]
    train_set = train_data[val_size:]
    
    os.makedirs("data/instruction_v1", exist_ok=True)
    with open("data/instruction_v1/train.jsonl", "w") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")
            
    with open("data/instruction_v1/val.jsonl", "w") as f:
        for item in val_set:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(train_set)} train and {len(val_set)} val examples.")

if __name__ == "__main__":
    build_dataset()
