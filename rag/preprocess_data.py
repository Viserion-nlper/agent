from glob import glob
text_lines = []
for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:        
        file_text = file.read()    
        text_lines += file_text.split("# ")