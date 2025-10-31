import os

def print_directory_tree(startpath, indent=""):
    """
    Recursively prints the structure of directories and files starting from 'startpath'.
    """
    items = sorted(os.listdir(startpath))
    for index, item in enumerate(items):
        path = os.path.join(startpath, item)
        connector = "└── " if index == len(items) - 1 else "├── "
        print(indent + connector + item)
        if os.path.isdir(path):
            extension = "    " if index == len(items) - 1 else "│   "
            print_directory_tree(path, indent + extension)

if __name__ == "__main__":
    # Change '.' to any directory path you want to explore
    root_dir = "."
    print(f"📂 Directory structure for: {os.path.abspath(root_dir)}\n")
    print_directory_tree(root_dir)
