import os
import re


def rename_test_files(directory):
    """
    Renames test files in the given directory from pattern 'test_name.py' to 'test_on_name.py'

    Args:
        directory (str): The directory containing the test files to rename

    Returns:
        dict: A mapping of old filenames to new filenames for successfully renamed files
    """
    # Store the mapping of old to new names
    renamed_files = {}

    # Regular expression to match test files
    pattern = r"^test_([^on_].+)\.py$"

    # List all files in the directory
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)

        # Check if the file matches our pattern and doesn't already have 'on' in it
        if match and "test_on_" not in filename:
            base_name = match.group(1)
            new_filename = f"test_{base_name}_model.py"

            # Construct full file paths
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            # Check if the new filename already exists
            if os.path.exists(new_path):
                print(f"Warning: {new_filename} already exists, skipping {filename}")
                continue

            try:
                # Rename the file
                os.rename(old_path, new_path)
                renamed_files[filename] = new_filename
                print(f"Renamed: {filename} -> {new_filename}")
            except OSError as e:
                print(f"Error renaming {filename}: {e}")

    return renamed_files


# Example usage
if __name__ == "__main__":
    # Get the current directory or specify your test directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print("Starting file rename operation...")
    renamed = rename_test_files(current_dir)

    print("\nSummary of renamed files:")
    if renamed:
        for old_name, new_name in renamed.items():
            print(f"- {old_name} → {new_name}")
    else:
        print("No files were renamed.")
