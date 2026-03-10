import os
import re
import sys
from pathlib import Path

from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config

class DataIngestor:
    def __init__(self):

        self.data_path = Config.DATA_PATH

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def clean_text(self, text):
        """Removes Markdown frontmatter, Hugo shortcodes, and extra formatting."""
        # 1. Remove specific frontmatter patterns found in GitLab docs
        text = re.sub(r'title:.*?\n', '', text)
        text = re.sub(r'controlled_document:.*?\n', '', text)
        text = re.sub(r'tags:.*?\n', '', text)
        text = re.sub(r'-\s+security_policy.*?\n', '', text)
        
        # 2. Remove Hugo shortcodes like {{< label ... >}}
        text = re.sub(r'\{\{<.*?>\}\}', '', text)
        
        # 3. Clean up leading/trailing whitespace and excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def extract_title(self, content):
        """Extracts title from the raw content before it's cleaned."""
        match = re.search(r'title:\s*"(.*?)"', content)
        if match:
            return match.group(1)
        return "Unknown Policy"

    def load_and_split(self):
        print(f"--- Loading and Cleaning policies from: {self.data_path} ---")
        
        loader = DirectoryLoader(
            self.data_path,
            glob="*.md",
            loader_cls=UnstructuredMarkdownLoader
        )
        
        raw_docs = loader.load()
        processed_chunks = []

        for doc in raw_docs:
            # Extract Title
            policy_title = self.extract_title(doc.page_content)
            
            # Clean content
            cleaned_content = self.clean_text(doc.page_content)
            doc.page_content = cleaned_content
            
            # Split into chunks
            doc_chunks = self.splitter.split_documents([doc])
            
            for chunk in doc_chunks:
                # Normalize the source path (Convert \ to /)
                normalized_source = chunk.metadata["source"].replace("\\", "/")
                
                chunk.metadata["policy_title"] = policy_title
                chunk.metadata["filename"] = os.path.basename(normalized_source)
                chunk.metadata["source"] = normalized_source 
                processed_chunks.append(chunk)

        print(f"--- Processed {len(raw_docs)} docs into {len(processed_chunks)} cleaned chunks. ---")
        return processed_chunks


def main() -> None:
    ingestor = DataIngestor()
    chunks = ingestor.load_and_split()
    if chunks:
        print("--- Sample chunk metadata ---")
        print(chunks[0].metadata)


if __name__ == "__main__":
    main()
