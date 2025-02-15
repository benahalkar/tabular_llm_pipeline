import os
import sys
import shutil
import numpy as np
import pandas as pd
import json

UIDS = set()

if len(sys.argv) != 5:
    sys.exit("Provide NUM_SAMPLES TRAIN_TEST_SPLIT ROWS COLUMNS in cmd line args")
else:
    print(sys.argv)

try:
    NUM_SAMPLES = int(sys.argv[1])
except Exception:
    NUM_SAMPLES = 100

try:
    TRAIN_TEST_SPLIT = float(sys.argv[2])
except Exception:
    TRAIN_TEST_SPLIT = 0.5

try:
    ROWS = int(sys.argv[3])
except Exception:
    ROWS = 10

try:
    COLUMNS = int(sys.argv[4])
except Exception:
    COLUMNS = 100


def beautify_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4, sort_keys=True)



def delete_all_contents(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return
    
    # Iterate through the folder and delete each item
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    first_path = os.path.join(folder_path, "Data") 
    os.makedirs(first_path)

    second_path = os.path.join(first_path, "Preprocessed")
    os.makedirs(second_path)


def generate_UID():
    uid = None

    while True:
        uid = np.random.randint(10000, 99999)
        if uid not in UIDS:
            UIDS.add(uid)
            break
    
    return uid


def return_single_chat():
    return [
        {
            "from": "human",
            "value": "what is the class?"
        },
        {
            "from": "gpt",
            "value": str(np.random.randint(0, 10))
        }
    ]



def generate_v0(json_filepath, folder, num_samples=1000):
    
    with open(json_filepath, "w") as f:
        f.write("[]")

    for sample in range(num_samples):

        random_id = generate_UID()
        csv_filename = f"{random_id}.csv"

        rows, columns = np.random.randint(10, 20), np.random.randint(5, 10)
        csv_data = np.random.rand(ROWS, COLUMNS)
        csv_data = np.round(csv_data, 2)
        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(folder, csv_filename), index=False, header=False)

        conversation = []

        new_dict = {
            "id": random_id,
            "table": csv_filename,
            "conversations": None
        }

        num_conv = np.random.randint(1, 5)
        for _ in range(num_conv):
            conversation += return_single_chat()
        
        new_dict["conversations"] = conversation
        new_dict["conversations"][0]["value"] = "<table>\n" + new_dict["conversations"][0]["value"]
        
        if os.path.getsize(json_filepath) > 2:
            with open(json_filepath, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                f.truncate()
                f.write(b',')
                f.write(json.dumps(new_dict).encode())
                f.write(b']')
                f.close()

        else:
            with open(json_filepath, "w") as f:
                json.dump([new_dict], f)
                f.close()




if __name__ == "__main__":

    data_folder = "./data"

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    delete_all_contents(data_folder)
    
    folder_name = os.path.join(data_folder, "Data", "Preprocessed")
    
    train_samples = int(NUM_SAMPLES * TRAIN_TEST_SPLIT)

    train_json_filename = "train_config.json"
    train_json_filepath = os.path.join(data_folder, train_json_filename)

    if os.path.exists(train_json_filename):
        os.remove(train_json_filename)
    
    generate_v0(train_json_filepath, folder_name, train_samples)

    beautify_json(train_json_filepath)
    
    eval_samples = NUM_SAMPLES - train_samples

    eval_json_filename = "eval_config.json"
    eval_json_filepath = os.path.join(data_folder, eval_json_filename)
    
    if os.path.exists(eval_json_filename):
        os.remove(eval_json_filename)
    
    generate_v0(eval_json_filepath, folder_name, eval_samples)

    beautify_json(eval_json_filepath)

    print(f"Generated {NUM_SAMPLES} samples")