from atlassian import Confluence
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import os

class ConfluenceLoader:
    def __init__(self, url, username, api_token):
        """
        Initialize the Confluence loader.
        
        Args:
            url: Your Confluence instance URL (e.g., 'https://your-domain.atlassian.net')
            username: Your Confluence username (typically your email)
            api_token: Your Confluence API token
        """
        self.confluence = Confluence(
            url=url,
            username=username,
            password=api_token,
            cloud=True  # Set to False if using server version
        )
        self.url = url
    
    def load_page(self, page_id):
        """Load a single page by ID."""
        page = self.confluence.get_page_by_id(page_id, expand='body.storage')
        
        # Extract HTML content
        html_content = page['body']['storage']['value']
        
        # Clean and process the HTML content
        text_content = self._clean_confluence_content(html_content)
        
        # Create a Document with metadata
        metadata = {
            'source': f"confluence_page_{page_id}",
            'title': page['title'],
            'url': self.url + page['_links']['webui'],
            'last_modified': page['version']['when'],
            'author': page['version']['by']['displayName']
        }
        
        return Document(page_content=text_content, metadata=metadata)
    
    def load_space(self, space_key, limit=50):
        """Load all pages in a Confluence space."""
        documents = []
        
        # Get all pages in the space
        pages = self.confluence.get_all_pages_from_space(space_key, limit=limit)
        
        for page in pages:
            try:
                doc = self.load_page(page['id'])
                documents.append(doc)
            except Exception as e:
                print(f"Error loading page {page['id']}: {e}")
        
        return documents

    def load_page_with_attachments(self, page_id):
        """Load a page and its attachments."""
        documents = []
        
        # Load the main page
        page_doc = self.load_page(page_id)
        documents.append(page_doc)
        
        # Get attachments
        attachments = self.confluence.get_attachments_from_content(page_id)
        
        for attachment in attachments:
            # Only process document attachments
            filename = attachment['title']
            if filename.endswith(('.pdf', '.docx', '.txt')):
                # Download the attachment
                content = self.confluence.download_attachment(attachment['id'])
                
                # Save temporarily
                temp_path = f"temp_{attachment['id']}"
                with open(temp_path, 'wb') as f:
                    f.write(content)
                
                # Use appropriate loader based on file type
                attachment_docs = []
                try:
                    if filename.endswith('.pdf'):
                        from langchain.document_loaders import PyPDFLoader
                        loader = PyPDFLoader(temp_path)
                        attachment_docs = loader.load()
                    elif filename.endswith('.docx'):
                        from langchain.document_loaders import UnstructuredWordDocumentLoader
                        loader = UnstructuredWordDocumentLoader(temp_path)
                        attachment_docs = loader.load()
                    elif filename.endswith('.txt'):
                        from langchain.document_loaders import TextLoader
                        loader = TextLoader(temp_path)
                        attachment_docs = loader.load()
                    
                    # Add metadata
                    for doc in attachment_docs:
                        doc.metadata['source'] = f"attachment_{attachment['id']}"
                        doc.metadata['filename'] = filename
                        doc.metadata['page_id'] = page_id
                    
                    documents.extend(attachment_docs)
                except Exception as e:
                    print(f"Error processing attachment {filename}: {e}")
                finally:
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        return documents
    
    def _clean_confluence_content(self, html_content):
        """Clean and process Confluence HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Handle tables - convert to text representation
        for table in soup.find_all('table'):
            table_text = "TABLE:\n"
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                row_text = " | ".join([cell.get_text(strip=True) for cell in cells])
                table_text += row_text + "\n"
            # Replace table with text representation
            table.replace_with(soup.new_string(table_text))
        
        # Remove expand macros but keep their content
        for expand in soup.find_all('ac:structured-macro', {'ac:name': 'expand'}):
            expand_content = expand.find('ac:rich-text-body')
            if expand_content:
                expand.replace_with(expand_content)
        
        # Extract text with better formatting
        text = ""
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
            if element.name.startswith('h'):
                level = int(element.name[1])
                prefix = '#' * level + ' '
                text += f"\n{prefix}{element.get_text()}\n"
            else:
                text += element.get_text() + "\n"
        
        # If no structured elements found, get all text
        if not text.strip():
            text = soup.get_text(separator='\n')
        
        return text
