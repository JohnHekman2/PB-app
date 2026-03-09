import os
import sys

# Ensure the root directory is in sys.path so we can import services
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.pandoc_converter import convert_md_to_docx_pandoc

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    md_file = os.path.join(base_dir, "tests", "markdown_test.md")
    template_file = os.path.join(base_dir, "wbtemplate.docx")
    output_dir = os.path.join(base_dir, "tests", "outputs")
    output_file = os.path.join(output_dir, "test_output.docx")
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found.")
        return
        
    if not os.path.exists(template_file):
        print(f"Error: {template_file} not found.")
        return
        
    with open(md_file, "r", encoding="utf-8") as f:
        markdown_content = f.read()
        
    print(f"Read {len(markdown_content)} characters from {md_file}")
    print("Converting...")
    
    docx_buffer = convert_md_to_docx_pandoc(markdown_content, template_file)
    
    with open(output_file, "wb") as f:
        f.write(docx_buffer.getvalue())
        
    print(f"Successfully wrote output to {output_file}")
    print(f"Output file size: {os.path.getsize(output_file)} bytes")

if __name__ == "__main__":
    main()
