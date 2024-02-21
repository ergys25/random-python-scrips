import os
import shutil


def organize_files_by_extension(folder_path):
    """
    Organizes files in a folder by their file extension.

    Parameters:
    folder_path (str): The path to the folder containing the files.

    Returns:
    None
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Get all files in the folder
    for item in os.listdir(folder_path):
        # Get full item path
        item_full_path = os.path.join(folder_path, item)

        # Check if it is a file
        if os.path.isfile(item_full_path):
            # Extract file extension
            file_extension = item.split(".")[-1].lower()
            if len(item.split(".")) < 2:
                continue  # Skip files without an extension

            # Create a folder for the extension if it doesn't exist
            extension_folder_path = os.path.join(folder_path, file_extension)
            if not os.path.exists(extension_folder_path):
                os.makedirs(extension_folder_path)

            try:
                # Move the file to the new folder
                shutil.move(item_full_path, extension_folder_path)
            except PermissionError as e:
                print(f"Failed to move file: {item_full_path}")
                print(f"Error: {e}")

    print(f"Files in {folder_path} have been organized by their file extension.")


if __name__ == "__main__":
    # Specify the folder to organize
    folder_to_organize = "Path to the folder to organize"
    organize_files_by_extension(folder_to_organize)
