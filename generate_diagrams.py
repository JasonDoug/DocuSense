import os
from PIL import Image, ImageDraw, ImageFont

# Create output directory
os.makedirs("docs", exist_ok=True)

# Helper for rounded rectangle with border and text
def draw_rounded_card(draw, box, bg_color, border_color, title, subtitle="", radius=12, border_width=2, title_color=(15, 23, 42), sub_color=(100, 116, 139), font_size=16):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=bg_color, outline=border_color, width=border_width)
    
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", max(11, font_size - 4))
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Draw Title
    if title:
        draw.text((x0 + 16, y0 + 12), title, fill=title_color, font=font_title)
    if subtitle:
        draw.text((x0 + 16, y0 + 36), subtitle, fill=sub_color, font=font_sub)

def draw_arrow(draw, start, end, color=(59, 130, 246), width=3, label=""):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    
    # Arrowhead
    if x1 == x2: # Vertical arrow down
        draw.polygon([(x2 - 6, y2 - 10), (x2 + 6, y2 - 10), (x2, y2)], fill=color)
    elif y1 == y2: # Horizontal arrow right
        draw.polygon([(x2 - 10, y2 - 6), (x2 - 10, y2 + 6), (x2, y2)], fill=color)

    if label:
        try:
            font_lbl = ImageFont.truetype("DejaVuSans.ttf", 11)
        except Exception:
            font_lbl = ImageFont.load_default()
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        draw.text((mid_x + 6, mid_y - 14), label, fill=(71, 85, 105), font=font_lbl)


# ==========================================
# 1. GENERATE FLOWCHART DIAGRAM
# ==========================================
def create_flowchart():
    width, height = 1000, 750
    img = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    # Main Title Header
    draw.text((40, 30), "DocuSense - PDF Processing Architecture Workflow", fill=(15, 23, 42), font=title_font)
    draw.text((40, 60), "End-to-End multi-model extraction, OCR fallback, and hierarchical summarization flow", fill=(100, 116, 139), font=sub_font)

    # 1. Input Node
    draw_rounded_card(draw, (40, 110, 260, 180), (255, 255, 255), (203, 213, 225), "PDF Document Input", "Upload / URL / Base64", title_color=(30, 41, 59))
    draw_arrow(draw, (150, 180), (150, 220), label="Submit")

    # 2. Pre-Analysis Node
    draw_rounded_card(draw, (40, 220, 260, 290), (239, 246, 255), (147, 197, 253), "PyMuPDF Pre-Analysis", "Detect layout, images, headers", title_color=(30, 58, 138))
    draw_arrow(draw, (150, 290), (150, 340), label="Route Pages")

    # 3. Decision Node / Splitter
    draw_rounded_card(draw, (40, 340, 260, 410), (243, 244, 246), (209, 213, 219), "Content Router", "Text Page vs Visual Page", title_color=(17, 24, 39))

    # Branch A: Text Page Path
    draw_arrow(draw, (260, 375), (360, 375), label="Text Content")
    draw_rounded_card(draw, (360, 335, 620, 415), (239, 246, 255), (59, 130, 246), "Text LLM Provider", "Gemini / OpenRouter / Ollama / OpenAI", title_color=(29, 78, 216))

    # Branch B: Image Page Path
    draw_arrow(draw, (150, 410), (150, 470), label="Visual Content")
    draw_rounded_card(draw, (40, 470, 300, 550), (250, 245, 255), (192, 132, 252), "Vision VLM Provider", "Process chart, diagram, or scan", title_color=(126, 34, 206))

    # Branch C: OCR Fallback
    draw_arrow(draw, (300, 510), (360, 510), label="If VLM Unavailable / Fails", color=(217, 119, 6))
    draw_rounded_card(draw, (360, 470, 620, 550), (254, 243, 199), (245, 158, 11), "Tesseract OCR Fallback", "Extract raw text & send to LLM", title_color=(146, 64, 14))

    # Merge Arrow to Summarizer
    draw_arrow(draw, (620, 375), (700, 375))
    draw_arrow(draw, (620, 510), (700, 510))
    draw.line([(700, 375), (700, 510)], fill=(59, 130, 246), width=3)
    draw_arrow(draw, (700, 442.5), (750, 442.5), label="Page Summaries")

    # 4. Hierarchical Summarizer
    draw_rounded_card(draw, (750, 400, 960, 485), (236, 253, 245), (52, 211, 153), "Hierarchical Summarizer", "Synthesize Section & Doc Summary", title_color=(6, 78, 59))

    # Output Arrow
    draw_arrow(draw, (855, 485), (855, 570), label="Final Output")

    # 5. Output Node
    draw_rounded_card(draw, (720, 570, 980, 670), (15, 23, 42), (51, 65, 85), "Structured Output & UI", "Markdown, Page Breakdown, Badges", title_color=(255, 255, 255), sub_color=(148, 163, 184))

    img.save("docs/flowchart.png")
    print("Generated docs/flowchart.png")


# ==========================================
# 2. GENERATE SEQUENCE DIAGRAM
# ==========================================
def create_sequence_diagram():
    width, height = 1050, 700
    img = Image.new("RGB", (width, height), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 13)
        font_box = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
        font_msg = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
        font_box = ImageFont.load_default()
        font_msg = ImageFont.load_default()

    draw.text((40, 30), "DocuSense - Component Sequence Diagram", fill=(15, 23, 42), font=title_font)
    draw.text((40, 60), "Real-time execution sequence between Client UI, FastAPI Server, PDF Engine, and AI Providers", fill=(100, 116, 139), font=sub_font)

    # Actor Lifelines
    actors = [
        {"name": "Client / Web UI", "x": 100},
        {"name": "FastAPI Server", "x": 320},
        {"name": "PDFParser Engine", "x": 540},
        {"name": "LLM / VLM Provider", "x": 760},
        {"name": "Tesseract OCR", "x": 950}
    ]

    y_top = 110
    y_bottom = 650

    # Draw Lifelines & Header Boxes
    for a in actors:
        x = a["x"]
        draw.line([(x, y_top + 40), (x, y_bottom)], fill=(203, 213, 225), width=2)
        draw.rounded_rectangle([x - 70, y_top, x + 70, y_top + 38], radius=8, fill=(15, 23, 42), outline=(51, 65, 85))
        draw.text((x - 60, y_top + 10), a["name"], fill=(255, 255, 255), font=font_box)

    # Helper for sequence arrows
    def seq_msg(from_idx, to_idx, y, text, is_return=False, color=(30, 41, 59)):
        x1 = actors[from_idx]["x"]
        x2 = actors[to_idx]["x"]
        
        if is_return:
            # Dashed line representation
            draw.line([(x1, y), (x2, y)], fill=color, width=2)
        else:
            draw.line([(x1, y), (x2, y)], fill=color, width=2)

        # Arrowhead
        if x1 < x2:
            draw.polygon([(x2 - 8, y - 5), (x2 - 8, y + 5), (x2, y)], fill=color)
        else:
            draw.polygon([(x2 + 8, y - 5), (x2 + 8, y + 5), (x2, y)], fill=color)

        mid_x = (x1 + x2) / 2
        draw.text((mid_x - 80, y - 16), text, fill=color, font=font_msg)

    # Sequence Messages
    seq_msg(0, 1, 180, "1. POST /api/parse-pdf-stream")
    seq_msg(1, 2, 230, "2. Initialize PDFParser(providers, options)")
    seq_msg(2, 2, 280, "3. Extract pages & detect images/layout", color=(71, 85, 105))
    seq_msg(2, 1, 330, "4. SSE Stream Log: 'Processing Page 1/N...'", is_return=True, color=(37, 99, 235))
    seq_msg(1, 0, 350, "5. Render Live Log in Terminal Console", is_return=True, color=(37, 99, 235))
    seq_msg(2, 3, 410, "6. process_text / process_image Prompt")
    seq_msg(3, 2, 460, "7. Return Clean Text & Page Summary", is_return=True)
    seq_msg(2, 4, 510, "8. OCR Fallback (if VLM unavailable/fails)", color=(217, 119, 6))
    seq_msg(4, 2, 550, "9. Return OCR Extracted Text", is_return=True, color=(217, 119, 6))
    seq_msg(2, 1, 600, "10. Deliver Final JSON Payload (data + metadata)", is_return=True, color=(5, 150, 105))
    seq_msg(1, 0, 630, "11. Render Markdown & Indicator Badges in UI", is_return=True, color=(5, 150, 105))

    img.save("docs/sequence-diagram.png")
    print("Generated docs/sequence-diagram.png")

if __name__ == "__main__":
    create_flowchart()
    create_sequence_diagram()
