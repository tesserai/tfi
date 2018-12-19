import re
from bs4 import BeautifulSoup
from bs4.element import Comment

def visible_text_for(html):
    def visible(element):
        if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
            return False
        elif isinstance(element, Comment):
            return False

        # Hidden if any parent has inline style "display: none"
        parent = element.parent
        while parent:
            if 'style' in parent.attrs and re.match('display:\\s*none', parent['style']):
                return False
            parent = parent.parent
        return True
    
    soup = BeautifulSoup(html, "html.parser")
    data = soup.findAll(text=True)
    return re.sub('\\s+', ' ', " ".join(t.strip() for t in filter(visible, data)))
