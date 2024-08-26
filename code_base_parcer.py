import os

def collect_codebase_contents(root_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_dir)
                
                outfile.write(f"\n\n--- File: {relative_path} ---\n\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                except UnicodeDecodeError:
                    outfile.write(f"[Unable to read file: {relative_path} - It may be a binary file.]\n")
                except Exception as e:
                    outfile.write(f"[Error reading file: {relative_path} - {str(e)}]\n")

def main():
    codebase_path = input("Enter the path to your codebase: ")
    output_file = 'codebase_contents.txt'
    
    collect_codebase_contents(codebase_path, output_file)
    
    print(f"Codebase contents written to {output_file}")

if __name__ == "__main__":
    main()