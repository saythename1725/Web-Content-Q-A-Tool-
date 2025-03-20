import requests
from bs4 import BeautifulSoup
from typing import List, Dict

class URLProcessor:
    def __init__(self):
        """Initialize URL processor with empty storage."""
        self._content_store = {}

    def process_urls(self, urls: List[str]) -> dict:
        """Process a list of URLs and store their content."""
        results = {}
        for url in urls:
            try:
                content = self._fetch_and_extract_content(url)
                if content:
                    self._content_store[url] = content
                    results[url] = "success"
                else:
                    results[url] = "failed - no content extracted"
            except Exception as e:
                results[url] = f"failed - {str(e)}"
        return results

    def _fetch_and_extract_content(self, url: str) -> str:
        """Fetch and extract main content from a URL."""
        try:
            print(f"\nFetching content from: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an error for bad responses
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Preserve the original soup before modifications
            original_soup = soup
            # Remove only scripts and styles initially
            for element in soup(['script', 'style']):
                element.decompose()

            # Special handling for Wikipedia
            if 'wikipedia.org' in url:
                # Get the page title
                title = soup.find(id='firstHeading')
                title_text = f"{title.get_text().strip()}. " if title else ""

                # Find the main content area
                main_content = soup.find(id='mw-content-text')
                if not main_content:
                    return ""

                # Clean up the content first
                for element in main_content.find_all(['sup', 'span']):
                    if any(x in str(element.get('class', [])) for x in ['reference', 'edit']):
                        element.decompose()

                # Remove unwanted elements
                for element in main_content.find_all(['table', 'div']):
                    classes = str(element.get('class', []))
                    if any(x in classes for x in ['reference', 'reflist', 'navbox', 'vertical-navbox']):
                        element.decompose()

                # Get all paragraphs first
                all_paragraphs = main_content.find_all('p', recursive=True)
                lead_content = []
                
                # Process first few paragraphs as lead content
                for i, p in enumerate(all_paragraphs):
                    if i < 3:  # Take first 3 paragraphs for lead
                        text = p.get_text().strip()
                        if len(text.split()) > 3:  # More lenient length check
                            lead_content.append(text)
                    else:
                        break

                # Get section content
                sections = []
                current_section = ""
                current_heading = ""
                
                for elem in main_content.children:
                    if elem.name in ['h2', 'h3']:
                        # Save previous section if exists
                        if current_section and current_heading:
                            sections.append(f"{current_heading}. {current_section}")
                        
                        # Start new section
                        heading_text = elem.get_text().strip()
                        if not any(x in heading_text.lower() for x in ['references', 'see also', 'external links', 'notes', 'citations']):
                            current_heading = heading_text
                            current_section = ""
                        else:
                            current_heading = ""
                    
                    elif current_heading and elem.name == 'p':
                        text = elem.get_text().strip()
                        if len(text.split()) > 3:
                            current_section += " " + text

                # Add final section if exists
                if current_section and current_heading:
                    sections.append(f"{current_heading}. {current_section}")
            else:
                # Find the main content area with special handling for technical content
                main_content = None
                
                # GeeksForGeeks specific handling
                if 'geeksforgeeks.org' in url:
                    main_content = soup.find('article', {'class': 'content'})
                    if not main_content:
                        main_content = soup.find('div', {'class': 'article-body'})
                
                # General fallback
                if not main_content:
                    main_content = (
                        soup.find(['article', 'main']) or 
                        soup.find('div', {'class': ['content', 'main', 'article', 'post-content']}) or
                        soup.find('div', {'id': ['content', 'main', 'article', 'post-content']}) or
                        soup
                    )
            
            # Extract text content with enhanced handling for technical content
            paragraphs = []
            code_blocks = []
            
            # First get all paragraph text and code blocks
            for element in main_content.find_all(['p', 'pre', 'code', 'div'], recursive=True):
                # Handle code blocks
                if element.name in ['pre', 'code'] or 'code' in element.get('class', []):
                    code_text = element.get_text().strip()
                    if code_text and len(code_text.split()) > 3:
                        code_blocks.append(f"Code example: {code_text}")
                # Handle text paragraphs
                elif element.name == 'p':
                    text = element.get_text().strip()
                    if text and len(text.split()) > 5:
                        paragraphs.append(text)
                # Handle div elements that might contain technical explanations
                elif element.name == 'div' and any(cls in str(element.get('class', [])) for cls in ['explanation', 'description', 'note', 'algorithm']):
                    text = element.get_text().strip()
                    if text and len(text.split()) > 5:
                        paragraphs.append(text)
            
            # Then get headers for context
            headers = []
            for h in main_content.find_all(['h1', 'h2', 'h3'], recursive=True):
                text = h.get_text().strip()
                if text and not any(x in text.lower() for x in ['references', 'see also', 'external links']):
                    headers.append(text)
            
            # Combine all content with fallback options
            if 'wikipedia.org' in url:
                # For Wikipedia, try structured content first
                content_parts = [title_text] + lead_content + sections
                content = ' '.join(part for part in content_parts if part)
                
                # Fallback to all paragraphs if structured content is too small
                if len(content.split()) < 100:
                    print("Wikipedia content too small, falling back to raw paragraphs")
                    content = ' '.join(p.get_text().strip() for p in main_content.find_all('p') 
                                    if len(p.get_text().strip().split()) > 3)
            else:
                # For non-Wikipedia, combine headers, paragraphs, and code blocks
                content = ' '.join(headers + paragraphs + code_blocks)
                
            # Fallback to basic text if content is still too small
            if len(content.split()) < 50:
                print("Content too small, extracting all text")
                content = ' '.join(t.strip() for t in soup.stripped_strings 
                                if len(t.strip()) > 0)
            
            # Enhanced content cleaning and validation
            content = content.replace('\n', ' ').replace('\t', ' ')  # Basic cleanup
            content = ' '.join(content.split())  # Normalize whitespace
            content = content.replace('"', '"').replace('"', '"')  # Fix quotes
            content = content.replace(''', "'").replace(''', "'")  # Fix apostrophes
            print(f"\nCleaned content sample: {content[:200]}...")
            
            # Validate minimum content
            if len(content.split()) < 10:
                print("Warning: Extracted content too short or empty")
                return ""
            
            print(f"Extracted content length: {len(content)} characters")
            if content:
                print(f"Sample content: {content[:200]}...")
            
            return content
            
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            return ""

    def get_stored_content(self) -> dict:
        """Return all stored content."""
        return self._content_store.copy()