
def get_text_and_last_paragraph(text):
    """
    Extracts the text and the last paragraph from the given text. 
    The last paragraph is defined as the text after the last double newline character.

    Args:
        text (str): The input text from which to extract the last paragraph.

    Returns:
        tuple: A tuple containing the cleaned text and the last paragraph.
    """
    cleaned_text = text.strip()
    paragraphs = cleaned_text.split('\n\n')
    last_para = paragraphs[-1].strip() if paragraphs else ""
    return [cleaned_text, last_para]


def parse_page_selection(selection_str: str, total_pages: int):
    """
    Parse page selection string (1-indexed) into list of 0-indexed page numbers.
    Supports formats: "1-5", "1,3,5", "1-3, 7, 10-12", "5-" (5 to end), "-5" (1 to 5).
    """
    if not selection_str or not str(selection_str).strip():
        return list(range(total_pages))

    pages = set()
    parts = str(selection_str).split(',')

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            subparts = part.split('-')
            if len(subparts) == 2:
                start_str, end_str = subparts[0].strip(), subparts[1].strip()
                try:
                    start = int(start_str) if start_str else 1
                except ValueError:
                    start = 1
                try:
                    end = int(end_str) if end_str else total_pages
                except ValueError:
                    end = total_pages

                start = max(1, min(start, total_pages))
                end = max(1, min(end, total_pages))

                if start <= end:
                    for p in range(start, end + 1):
                        pages.add(p - 1)
        else:
            try:
                p = int(part)
                if 1 <= p <= total_pages:
                    pages.add(p - 1)
            except ValueError:
                pass

    result = sorted(list(pages))
    return result if result else list(range(total_pages))

