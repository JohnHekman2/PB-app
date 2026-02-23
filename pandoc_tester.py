import pypandoc

try:
    print("Starting conversion...")
    
    # Convert the markdown file to a docx file
    pypandoc.convert_file(
        source_file='rapport_compleet_markdown.md', 
        to='docx', 
        outputfile='test_output.docx'
    )
    
    print("Success! Check your folder for test_output.docx")

except Exception as e:
    print(f"An error occurred: {e}")