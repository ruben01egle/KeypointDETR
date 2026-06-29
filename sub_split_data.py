import os
import random
import shutil

def downsample_stratified_to_folder(source_dir, output_dir, percentage, seed=42):
    """
    Reads train.txt and val.txt from source_dir, downsamples each class 
    proportionally using classID prefixes, and saves them as train.txt 
    and val.txt inside output_dir. Also copies the original test.txt.
    """
    # Create the target subdirectory for this scenario
    os.makedirs(output_dir, exist_ok=True)
    
    files_to_process = ['train.txt', 'val.txt']
    
    for filename in files_to_process:
        source_file = os.path.join(source_dir, filename)
        output_file = os.path.join(output_dir, filename)
        
        if not os.path.exists(source_file):
            print(f"Error: {source_file} does not exist!")
            continue

        with open(source_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Group lines by their class ID prefix (everything before the first '-')
        class_groups = {}
        for line in lines:
            if '-' in line:
                class_id = line.split('-')[0]
            else:
                class_id = "unknown"
                
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(line)

        # Downsample within each class group independently (identical logic)
        random.seed(seed)
        final_sub_lines = []
        
        print(f"Processing {filename} for {percentage}% scenario:")
        for class_id, class_lines in class_groups.items():
            total_class_count = len(class_lines)
            # Ensure at least 1 sample remains if the class has data
            sub_class_count = max(1, int(total_class_count * (percentage / 100.0)))
            
            sampled_lines = random.sample(class_lines, sub_class_count)
            final_sub_lines.extend(sampled_lines)
            print(f"  > Class {class_id}: Sampled {sub_class_count} out of {total_class_count}")

        # Sort alphabetically to keep the file clean and structured
        final_sub_lines.sort()

        # Save the stratified downsampled split with its standard name
        with open(output_file, 'w') as f:
            for line in final_sub_lines:
                f.write(f"{line}\n")
                
        print(f"Successfully created: {os.path.basename(output_dir)}/{filename} | Total: {len(final_sub_lines)}\n")

    # Copy the original test.txt into the folder to complete the split directory
    source_test = os.path.join(source_dir, 'test.txt')
    output_test = os.path.join(output_dir, 'test.txt')
    if os.path.exists(source_test):
        shutil.copyfile(source_test, output_test)
        print(f"Copied original test.txt to {os.path.basename(output_dir)}/ for the evaluation benchmark.\n")


if __name__ == "__main__":
    # Resolve relative paths dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_splits_dir = os.path.abspath(os.path.join(current_dir, "..", "KeypointNet", "splits"))
    
    # Target percentages for your paper's simulated scarcity scenarios
    percentage_steps = [50, 25, 10, 5]
    
    print("Simulating stratified data scarcity based on class prefixes (saving to folders)...")
    for step in percentage_steps:
        target_folder_name = f"split_{step}"
        target_folder_path = os.path.join(original_splits_dir, target_folder_name)
        
        print(f"================ Simulating {step}% Data Scenario ================")
        downsample_stratified_to_folder(original_splits_dir, target_folder_path, percentage=step, seed=42)
        
    print("All scenario folders successfully created inside your splits directory!")