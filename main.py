import json
import os
import fitz  # PyMuPDF
from PIL import Image
import io
import concurrent.futures
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass, field
import logging
from collections import Counter
import pytesseract
import requests
import tempfile
from utils.BaseProviders import BaseLLMProvider, BaseVLMProvider
from utils.OpenAIProviders import OpenAIProvider, OpenAIVisionProvider
from utils.helpers import parse_page_selection


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]')
logger = logging.getLogger(__name__)


# Define data structures

@dataclass
class ExtractedContent:
    """Class for storing extracted content and metadata for a page"""
    text: str  # Extracted text or image description
    summary: str  # Summary of the content
    page_num: int  # Original page number (0-indexed)
    has_images: bool = False  # True if this content came primarily from an image/VLM
    used_ocr_fallback: bool = False  # True if OCR fallback was used for this page
    extraction_method: str = "llm"  # "llm", "vlm", "ocr", or "raw_text"


# Removed unused class PDFProcessingChunk as processing unit is now page


class PDFParser:
    """PDF parsing class with LLM and VLM integration

    Processes PDF page by page, optionally in parallel, and uses LLM/VLM
    to extract and summarize content.

    Attributes:
        llm_provider (BaseLLMProvider): Language model provider
        vlm_provider (BaseVLMProvider): Vision model provider
        process_sequentially (bool): Whether to process pages sequentially
                                     (enables page-to-page summary context)
                                     or in parallel (faster, no page context).
        max_workers (int): Maximum number of workers for parallel processing
        min_image_size (int): Minimum size (width or height) of an image in pixels
                              to consider a page as having significant images.
        ocr_fallback (bool): Whether to use OCR as a fallback for image processing
                             if VLM is not available or fails.
        summarization_group_size (int): Number of pages to group together for
                                        hierarchical summarization steps.

    Methods:
        _detect_has_images: Detect if the PDF has images efficiently by sampling pages
        _extract_headers_footers: Attempt to identify headers and footers for removal
        _render_page_to_image: Render a PDF page to a PIL Image
        _process_text_page: Process a page of text using LLM
        _process_image_page: Process a page as image using VLM (or OCR fallback)
        _process_single_page: Helper to process a single page (text or image)
                              and apply header/footer removal.
        _hierarchical_summarize: Create a hierarchical summary of the document
        parse_pdf: Parse a PDF file or URL and extract content using LLM/VLM

    Example:
        >>> parser = PDFParser(llm_provider=OpenAIProvider(), vlm_provider=OpenAIVisionProvider())
        >>> result = parser.parse_pdf("example.pdf")
        >>> print(result)
        {'text': '...', 'pages': [...], 'summary': '...', 'metadata': {...}}
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider = None,
        vlm_provider: BaseVLMProvider = None,
        # Process pages sequentially (enables page context)
        process_sequentially: bool = False,
        max_workers: int = 4,
        min_image_size: int = 100,  # Min image size in pixels to consider page 'has_images'
        ocr_fallback: bool = True,
        summarization_group_size: int = 5,
    ):
        if not llm_provider and not vlm_provider and not ocr_fallback:
            logger.warning(
                "No LLM, VLM, or OCR fallback configured. Content extraction will be basic text/image placeholders.")

        self.llm_provider = llm_provider
        self.vlm_provider = vlm_provider
        self.process_sequentially = process_sequentially
        self.max_workers = max_workers
        self.min_image_size = min_image_size
        self.ocr_fallback = ocr_fallback
        self.summarization_group_size = summarization_group_size

        # Initialize fallback OCR if needed
        self.pytesseract = None
        if self.ocr_fallback:
            try:
                import pytesseract
                self.pytesseract = pytesseract
            except ImportError:
                logger.warning(
                    "Tesseract OCR not installed, OCR fallback disabled. Install with 'pip install pytesseract'")
                self.ocr_fallback = False
            except Exception as e:
                logger.warning(
                    f"Could not initialize Tesseract OCR: {e}", exc_info=True)
                self.ocr_fallback = False

    def _detect_has_images(self, doc, sample_size: int = 10) -> bool:
        """Detect if the PDF has images efficiently by sampling pages"""

        if len(doc) == 0:
            return False

        # If the document has few pages, check all
        if len(doc) <= sample_size:
            pages_to_check = range(len(doc))
        else:
            # Otherwise, check a sample of pages spread throughout the document
            # Ensure we don't go out of bounds
            stride = max(1, (len(doc) - 1) // (sample_size - 1)
                         ) if sample_size > 1 else 1
            pages_to_check = [min(i * stride, len(doc) - 1)
                              for i in range(sample_size)]
            # Remove duplicates and sort
            pages_to_check = sorted(list(set(pages_to_check)))

        logger.debug(f"Checking pages {pages_to_check} for images")

        for page_num in pages_to_check:
            try:
                page = doc[page_num]
                # Use full=False for speed if bbox not needed
                image_list = page.get_images(full=False)

                if image_list:
                    # Check dimensions of at least one image on the page
                    # More robust check would involve rendering or extracting image bytes,
                    # but get_images is faster. Assuming min_image_size check
                    # is a proxy for significant visual content vs decorative icons.
                    # A truly robust check might need rendering, but this is a fast heuristic.
                    for img_info in image_list:
                        # img_info is (xref, smask, width, height, bpc, filter, colorspace, ...)
                        # This info is available without extraction
                        width = img_info[2]
                        height = img_info[3]
                        if width is not None and height is not None and (width > self.min_image_size or height > self.min_image_size):
                            logger.debug(
                                f"Found image > {self.min_image_size}px on page {page_num}")
                            return True
                # If no image met the size threshold, continue checking other pages
                logger.debug(
                    f"Found images on page {page_num} but none exceeded {self.min_image_size}px")

            except Exception as e:
                logger.warning(
                    f"Error checking page {page_num} for images: {e}", exc_info=True)

        logger.debug("No significant images found across sampled pages.")
        return False

    def _extract_headers_footers(self, doc, sample_size: int = 10) -> Tuple[List[str], List[str]]:
        """
        Attempt to identify headers and footers for removal.
        Looks for text lines repeating at the very top or bottom of sampled pages.
        Returns (headers, footers) lists.

        Note: This is a heuristic and may not be perfect for complex layouts
        or variable header/footer content (like page numbers).
        """
        headers = []
        footers = []

        total_pages = len(doc)
        if total_pages < 3:  # Need at least 3 pages to detect patterns
            return [], []

        # Sample pages spread throughout the document
        sample_pages = min(sample_size, total_pages)
        stride = max(1, (total_pages - 1) // (sample_pages - 1)
                     ) if sample_pages > 1 else 1
        pages_to_check = [min(i * stride, total_pages - 1)
                          for i in range(sample_pages)]
        pages_to_check = sorted(list(set(pages_to_check)))

        logger.debug(f"Checking pages {pages_to_check} for headers/footers")

        # Get text from top and bottom of pages
        top_lines = []
        bottom_lines = []
        line_count = 0

        for page_num in pages_to_check:
            try:
                page = doc[page_num]
                # Extract text blocks and sort by y-coordinate
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda block: block[1])  # Sort by y0

                if blocks:
                    # Get lines from the top-most block(s)
                    # Consider blocks within 50 pixels of the very top block
                    top_y_threshold = blocks[0][1] + 50
                    current_top_lines = []
                    for block in blocks:
                        if block[1] <= top_y_threshold:
                            current_top_lines.extend(
                                [line.strip() for line in block[4].split('\n') if line.strip()])
                        else:
                            break  # Assume blocks are sorted, so we are past the top

                    if current_top_lines:
                        # Use only the very first line found
                        top_lines.append(current_top_lines[0])

                    # Get lines from the bottom-most block(s)
                    # Sort by y1 descending
                    blocks.sort(key=lambda block: block[3], reverse=True)
                    # Consider blocks within 50 pixels of the very bottom block
                    bottom_y_threshold = blocks[0][3] - 50
                    current_bottom_lines = []
                    for block in blocks:
                        if block[3] >= bottom_y_threshold:
                            current_bottom_lines.extend(
                                [line.strip() for line in block[4].split('\n') if line.strip()])
                        else:
                            break  # Assume blocks are sorted by y1 desc

                    if current_bottom_lines:
                        # Use only the very last line found
                        bottom_lines.append(current_bottom_lines[-1])
                    line_count += 1  # Increment line_count only if we successfully processed a page

            except Exception as e:
                logger.warning(
                    f"Error extracting lines from page {page_num}: {e}", exc_info=True)

        # Find repeating patterns in the collected lines
        if line_count == 0:
            return [], []  # No pages processed successfully

        top_counter = Counter(top_lines)
        bottom_counter = Counter(bottom_lines)

        # If a text appears in more than 70% of processed sample pages, consider it header/footer
        header_threshold = line_count * 0.7
        footer_threshold = line_count * 0.7

        for text, count in top_counter.items():
            if count >= header_threshold and text:
                headers.append(text)

        for text, count in bottom_counter.items():
            if count >= footer_threshold and text:
                footers.append(text)

        return headers, footers

    # Removed _extract_semantic_chunks

    def _render_page_to_image(self, page, dpi: int = 300) -> Image.Image:
        """Render a PDF page to a PIL Image"""
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img
        except Exception as e:
            logger.error(
                f"Failed to render page {page.number} to image: {e}", exc_info=True)
            return None

    def _process_text_page(self, page_text: str, page_num: int, prev_summary: str = None) -> ExtractedContent:
        """Process a page of text using LLM"""
        logger.debug(f"Processing text for page {page_num}")
        if not self.llm_provider:
            return ExtractedContent(
                text=page_text,
                summary=f"Page {page_num + 1} content (LLM not available)",
                page_num=page_num,
                has_images=False,
                used_ocr_fallback=False,
                extraction_method="raw_text"
            )

        try:
            # LLM provider's process_text is expected to handle summarization
            processed_text, summary = self.llm_provider.process_text(
                text=page_text, context=prev_summary)  # Pass prev_summary as context

            # Ensure text and summary are strings and not None
            processed_text = processed_text if isinstance(
                processed_text, str) else str(processed_text)
            summary = summary if isinstance(summary, str) else str(summary)

            return ExtractedContent(
                text=processed_text,
                summary=summary,
                page_num=page_num,
                has_images=False,
                used_ocr_fallback=False,
                extraction_method="llm"
            )
        except Exception as e:
            logger.error(
                f"LLM processing failed for page {page_num}: {e}", exc_info=True)
            return ExtractedContent(
                text=f"[Error processing text for page {page_num + 1}]",
                summary=f"Processing error on page {page_num + 1}",
                page_num=page_num,
                has_images=False,
                used_ocr_fallback=False,
                extraction_method="llm"
            )

    def _process_image_page(self, page, page_num: int, prev_summary: str = None) -> ExtractedContent:
        """Process a page as image using VLM (or OCR fallback)"""
        logger.debug(f"Processing image for page {page_num}")
        page_image = self._render_page_to_image(page)

        if page_image is None:
            logger.warning(
                f"Could not render page {page_num} for image processing.")
            return ExtractedContent(
                text=f"[Could not render image for page {page_num + 1}]",
                summary=f"Failed to process image on page {page_num + 1}",
                page_num=page_num,
                has_images=True,
                used_ocr_fallback=False,
                extraction_method="raw_text"
            )

        if self.vlm_provider:
            try:
                # VLM provider's process_image is expected to handle description and summarization
                extracted_text, summary = self.vlm_provider.process_image(
                    image=page_image, context=prev_summary)  # Pass prev_summary as context

                # Ensure text and summary are strings and not None
                extracted_text = extracted_text if isinstance(
                    extracted_text, str) else str(extracted_text)
                summary = summary if isinstance(summary, str) else str(summary)

                return ExtractedContent(
                    text=extracted_text,
                    # Use default summary if VLM returns None
                    summary=summary or f"Page {page_num + 1} content (VLM)",
                    page_num=page_num,
                    has_images=True,
                    used_ocr_fallback=False,
                    extraction_method="vlm"
                )
            except Exception as e:
                logger.error(
                    f"VLM processing failed for page {page_num}: {e}", exc_info=True)
                # Fall through to OCR or placeholder if VLM fails

        if self.ocr_fallback and self.pytesseract:
            logger.info(
                f"VLM not available or failed for page {page_num}, attempting OCR fallback.")
            try:
                extracted_text = self.pytesseract.image_to_string(page_image)
                summary = f"Page {page_num + 1} content (OCR fallback)"
                if self.llm_provider and extracted_text.strip():
                    logger.debug(
                        f"Sending OCR text from page {page_num} to LLM for summary")
                    try:
                        # Use LLM to get a better summary from the OCR text if available
                        _, llm_summary = self.llm_provider.process_text(
                            # Limit text sent to LLM
                            text=extracted_text[:4000], context=prev_summary)
                        summary = llm_summary or summary  # Use LLM summary if not empty
                    except Exception as llm_e:
                        logger.warning(
                            f"LLM summary failed for OCR text on page {page_num}: {llm_e}", exc_info=True)

                return ExtractedContent(
                    text=extracted_text,
                    summary=summary,
                    page_num=page_num,
                    has_images=True,  # Still mark as has_images as the source was an image
                    used_ocr_fallback=True,
                    extraction_method="ocr"
                )
            except Exception as e:
                logger.error(
                    f"OCR fallback failed for page {page_num}: {e}", exc_info=True)

        # No VLM and no OCR fallback
        logger.warning(
            f"No VLM or OCR fallback available for image page {page_num}.")
        return ExtractedContent(
            text=f"[Image content on page {page_num + 1}]",
            summary=f"Page {page_num + 1} contains image content",
            page_num=page_num,
            has_images=True,
            used_ocr_fallback=False,
            extraction_method="none"
        )

    def _process_single_page(self, page_num: int, doc, has_images: bool, headers: List[str], footers: List[str], prev_summary: str = None) -> ExtractedContent:
        """Helper function to process a single page (text or image)"""
        try:
            page = doc[page_num]
            page_has_images_flag = has_images  # Assume based on document-level detection

            # Refine page_has_images_flag based on actual page content if needed, e.g.
            # if not has_images_globally, but page.get_images() has large images.
            # For simplicity now, we rely on the document-level flag.

            if page_has_images_flag and (self.vlm_provider or self.ocr_fallback):
                # Process as image if global flag is set and we have VLM/OCR capability
                return self._process_image_page(page=page, page_num=page_num, prev_summary=prev_summary)
            else:
                # Process as text (either no global images detected, or no VLM/OCR for image pages)
                page_text = page.get_text()

                # Apply header/footer removal heuristic
                cleaned_text = page_text
                for header in headers:
                    cleaned_text = cleaned_text.replace(header, "")
                for footer in footers:
                    cleaned_text = cleaned_text.replace(footer, "")

                # Basic cleanup of excessive newlines after removal
                cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()

                return self._process_text_page(page_text=cleaned_text, page_num=page_num, prev_summary=prev_summary)

        except Exception as e:
            logger.error(
                f"Critical error processing page {page_num}: {e}", exc_info=True)
            return ExtractedContent(
                text=f"[Critical error processing page {page_num + 1}]",
                summary=f"Critical error on page {page_num + 1}",
                page_num=page_num,
                has_images=has_images
            )

    # Removed _merge_adjacent_chunks

    def _hierarchical_summarize(self, page_contents: List[ExtractedContent]) -> str:
        """Create a hierarchical summary of the document from page summaries"""

        if not page_contents:
            return ""

        if not self.llm_provider:
            # If no LLM, just concatenate page summaries
            return "\n\n".join([f"Page {pc.page_num+1} Summary: {pc.summary}" for pc in page_contents])

        # Ensure page_contents are sorted by page_num
        page_contents.sort(key=lambda x: x.page_num)

        # Group pages into manageable sizes for intermediate summarization
        group_size = self.summarization_group_size
        chunk_groups = [page_contents[i:i+group_size]
                        for i in range(0, len(page_contents), group_size)]
        group_summaries = []

        logger.info(
            f"Performing hierarchical summarization with group size {group_size}")

        for i, group in enumerate(chunk_groups):
            # Combine summaries and potentially text from the group for intermediate summary
            group_input_text = "\n\n".join(
                [f"Page {pc.page_num+1}: {pc.summary}" for pc in group if pc.summary]
            )

            # Only process if we have meaningful input
            if group_input_text.strip():
                logger.debug(f"Summarizing group {i+1}/{len(chunk_groups)}")
                prompt_template = """
                The following are summaries and key points from a section of a document (Pages {first_page}-{last_page}):

                {text}

                Please provide a cohesive summary that integrates these points (max 200 words). Focus on the main ideas and flow. Start directly with the summary. Do not include any other introductory or concluding text.
                """
                first_page = group[0].page_num + 1
                last_page = group[-1].page_num + 1

                try:
                    _, group_summary = self.llm_provider.process_text(
                        text=group_input_text,
                        prompt_template=prompt_template.format(
                            first_page=first_page, last_page=last_page, text="{text}")
                    )
                    if group_summary and group_summary.strip():
                        group_summaries.append(
                            f"Section Summary (Pages {first_page}-{last_page}):\n{group_summary}")
                except Exception as e:
                    logger.error(
                        f"Failed to summarize group {i+1} (Pages {first_page}-{last_page}): {e}", exc_info=True)
                    group_summaries.append(
                        f"Section Summary (Pages {first_page}-{last_page}): [Summarization failed]")

        # Summarize the group summaries to get the final document summary
        if not group_summaries:
            return "Could not generate summary."

        final_input_text = "\n\n".join(group_summaries)

        if len(group_summaries) > 1:
            logger.info(
                "Summarizing section summaries for final document summary.")
            prompt_template = """
            The following are summaries from different sections of a document:

            {text}

            Provide an integrated overall document summary combining these section summaries (max 300 words). Focus on the core topics, arguments, or information presented across the entire document. Start directly with the summary. Do not include any other introductory or concluding text.
            """
            try:
                _, document_summary = self.llm_provider.process_text(
                    text=final_input_text,
                    prompt_template=prompt_template.format(text="{text}")
                )
                return document_summary if document_summary and document_summary.strip() else "Could not generate final document summary."
            except Exception as e:
                logger.error(
                    f"Failed to generate final document summary: {e}", exc_info=True)
                return "Could not generate final document summary due to an error."
        else:
            # If only one group, its summary is the document summary
            # Remove the "Section Summary (Pages X-Y):" prefix
            return re.sub(r"^Section Summary \(Pages \d+-\d+\):\s*", "", group_summaries[0]).strip()

    def _notify(self, msg: str, level: str = "info", callback = None):
        if level == "error":
            logger.error(msg)
        elif level == "warning":
            logger.warning(msg)
        else:
            logger.info(msg)
        if callback:
            try:
                callback(msg, level)
            except Exception:
                pass

    def parse_pdf(self, pdf_input: str, page_selection: Optional[str] = None, status_callback = None) -> Dict:
        """
        Parse a PDF file or URL and extract content using LLM/VLM.

        Args:
            pdf_input: Path to the PDF file, URL to the PDF, or base64 string.
            page_selection: Optional page range/selection string (e.g. "1-5", "1,3,5", "10-20").
            status_callback: Optional callback function f(msg, level) for streaming real-time status logs.

        Returns:
            Dictionary with extracted content, page-by-page summaries,
            and overall document summary. Includes metadata.
            Returns {'error': ...} on failure.
        """
        temp_file = None
        doc = None
        pdf_path = None

        try:
            self._notify(f"Starting PDF parsing pipeline...", "info", status_callback)

            # 1. Handle input (URL, Base64, File Path)
            if pdf_input.startswith(('http://', 'https://')):
                self._notify(f"Downloading PDF from URL: {pdf_input[:80]}...", "info", status_callback)
                try:
                    response = requests.get(pdf_input, stream=True)
                    response.raise_for_status()
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, suffix='.pdf')
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    temp_file.close()
                    pdf_path = temp_file.name
                    self._notify("PDF downloaded successfully.", "info", status_callback)
                except requests.exceptions.RequestException as e:
                    self._notify(f"Failed to download PDF from URL: {e}", "error", status_callback)
                    raise ValueError(f"Failed to download PDF: {e}") from e
            elif pdf_input.startswith('data:application/pdf;base64,'):
                self._notify("Decoding Base64 PDF data...", "info", status_callback)
                try:
                    import base64
                    base64_data = pdf_input.split(',', 1)[1]
                    pdf_bytes = base64.b64decode(base64_data)
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, suffix='.pdf')
                    temp_file.write(pdf_bytes)
                    temp_file.close()
                    pdf_path = temp_file.name
                    self._notify("Decoded Base64 PDF file.", "info", status_callback)
                except Exception as e:
                    self._notify(f"Failed to decode base64 PDF: {e}", "error", status_callback)
                    raise ValueError(f"Failed to decode base64 PDF: {e}") from e
            elif os.path.exists(pdf_input):
                self._notify(f"Loading local PDF file...", "info", status_callback)
                pdf_path = pdf_input
            else:
                self._notify(f"Invalid input source: {pdf_input[:50]}", "error", status_callback)
                raise FileNotFoundError(f"Input is not a valid file path or URL: {pdf_input}")

            # 2. Open PDF document
            try:
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                if total_pages == 0:
                    self._notify("PDF document contains no pages.", "warning", status_callback)
                    return {"text": "", "pages": [], "summary": "Document is empty.", "metadata": {"total_pages": 0, "contains_images": False}}

                pages_to_process = parse_page_selection(page_selection, total_pages)
                self._notify(f"Opened PDF successfully ({total_pages} total pages). Processing {len(pages_to_process)} selected pages (Range: '{page_selection or 'All'}').", "info", status_callback)
            except fitz.FileDataError as e:
                self._notify(f"Failed to open PDF file: {e}", "error", status_callback)
                raise ValueError(f"Failed to open PDF file: {e}") from e
            except Exception as e:
                self._notify(f"Error opening PDF: {e}", "error", status_callback)
                raise RuntimeError(f"An unexpected error occurred while opening PDF: {e}") from e

            # 3. Pre-analysis: Detect images, headers, footers
            self._notify("Scanning document layout and inspecting visual content...", "info", status_callback)
            has_images = self._detect_has_images(doc)
            self._notify(f"Visual content scan completed (Has images: {has_images})", "info", status_callback)

            headers, footers = self._extract_headers_footers(doc)
            if headers or footers:
                self._notify(f"Detected repeating headers/footers for cleaning.", "info", status_callback)

            all_page_contents: List[ExtractedContent] = []

            # 4. Process pages (Sequential or Parallel)
            if self.process_sequentially:
                self._notify(f"Processing {len(pages_to_process)} selected pages sequentially with context chaining...", "info", status_callback)
                prev_summary = None
                for idx, page_num in enumerate(pages_to_process):
                    self._notify(f"Processing Page {page_num + 1} ({idx + 1}/{len(pages_to_process)})...", "info", status_callback)
                    page_content = self._process_single_page(
                        page_num=page_num,
                        doc=doc,
                        has_images=has_images,
                        headers=headers,
                        footers=footers,
                        prev_summary=prev_summary
                    )
                    all_page_contents.append(page_content)
                    prev_summary = page_content.summary
                    self._notify(f"Page {page_num + 1} complete ({idx + 1}/{len(pages_to_process)}). Method: {page_content.extraction_method.upper()}", "info", status_callback)

            else:
                self._notify(f"Processing {len(pages_to_process)} selected pages in parallel using {self.max_workers} worker threads...", "info", status_callback)
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_page = {
                        executor.submit(
                            self._process_single_page,
                            page_num=page_num,
                            doc=doc,
                            has_images=has_images,
                            headers=headers,
                            footers=footers,
                            prev_summary=None
                        ): page_num
                        for page_num in pages_to_process
                    }

                    # Collect results as they complete
                    results_dict: Dict[int, ExtractedContent] = {}
                    for future in concurrent.futures.as_completed(future_to_page):
                        page_num = future_to_page[future]
                        try:
                            page_content = future.result()
                            results_dict[page_num] = page_content
                            self._notify(f"Finished processing page {page_num + 1} (parallel).", "info", status_callback)
                        except Exception as e:
                            self._notify(f"Error processing page {page_num + 1} in parallel: {e}", "error", status_callback)
                            results_dict[page_num] = ExtractedContent(
                                text=f"[Error processing page {page_num + 1}]",
                                summary=f"Error on page {page_num + 1}",
                                page_num=page_num,
                                has_images=has_images
                            )

                    # Sort results by page number
                    all_page_contents = [results_dict[i]
                                         for i in sorted(results_dict.keys())]

            # 5. Generate document summary
            self._notify("Building hierarchical document summary...", "info", status_callback)
            document_summary = self._hierarchical_summarize(all_page_contents)
            self._notify("Generated document summary successfully.", "info", status_callback)

            # Check if OCR fallback was used on any page
            any_ocr_used = any(content.used_ocr_fallback for content in all_page_contents)

            llm_prov_name = getattr(self.llm_provider, "provider_type", self.llm_provider.__class__.__name__) if self.llm_provider else "None"
            llm_model_name = getattr(self.llm_provider, "model_name", "N/A") if self.llm_provider else "None"

            vlm_prov_name = getattr(self.vlm_provider, "provider_type", self.vlm_provider.__class__.__name__) if self.vlm_provider else "None"
            vlm_model_name = getattr(self.vlm_provider, "model_name", "N/A") if self.vlm_provider else "None"

            # 6. Format output
            result = {
                # Join text from all pages, maintaining page breaks
                "text": "\n\n" + "\n\n".join([
                    f"--- Page {content.page_num + 1} ---\n\n{content.text}"
                    for content in all_page_contents
                ]),
                "pages": [
                    {
                        "page_num": content.page_num + 1,
                        "text": content.text,
                        "summary": content.summary,
                        "has_images": content.has_images,
                        "used_ocr_fallback": content.used_ocr_fallback,
                        "extraction_method": content.extraction_method,
                    }
                    for content in all_page_contents
                ],
                "summary": document_summary,
                "metadata": {
                    "total_pages": total_pages,
                    "pages_processed": len(pages_to_process),
                    "page_selection": page_selection if page_selection else "all",
                    "contains_images": has_images,
                    "processed_sequentially": self.process_sequentially,
                    "ocr_fallback_used": any_ocr_used,
                    "llm_provider": llm_prov_name,
                    "llm_model": llm_model_name,
                    "vlm_provider": vlm_prov_name,
                    "vlm_model": vlm_model_name,
                }
            }
            logger.info("PDF parsing completed successfully.")
            return result

        except Exception as e:
            # Catch any exception not caught earlier
            logger.error(
                f"An unhandled error occurred during PDF processing: {e}", exc_info=True)
            return {"error": f"Failed to process PDF: {str(e)}"}

        finally:
            # 7. Cleanup temporary file and document handle
            if doc is not None:
                try:
                    doc.close()
                    logger.debug("Closed PDF document.")
                except Exception as e:
                    logger.warning(
                        f"Error closing PDF document: {e}", exc_info=True)
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                    logger.debug(f"Deleted temporary file: {temp_file.name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete temporary file {temp_file.name}: {e}", exc_info=True)


# Example usage function
def parse_pdf_with_custom_providers(
    pdf_url: str,
    llm_provider_name: str = "openai",
    vlm_provider_name: str = "openai",
    llm_api_key: str = None,
    vlm_api_key: str = None,
    parallel: bool = True,
    output_path: str = None
) -> Dict:
    """
    Parse a PDF with custom LLM and VLM providers

    Args:
        pdf_path: Path to the PDF file
        llm_provider_name: Name of the LLM provider ('openai', 'anthropic', None)
        vlm_provider_name: Name of the VLM provider ('openai', 'gemini', None)
        llm_api_key: API key for the LLM provider
        vlm_api_key: API key for the VLM provider
        parallel: Whether to process pages in parallel
        output_path: Path to save the output JSON file

    Returns:
        Dictionary with extracted content
    """

    # Initialize providers
    llm_provider = None
    vlm_provider = None

    if llm_provider_name == "openai":
        llm_provider = OpenAIProvider(api_key=llm_api_key)

    if vlm_provider_name == "openai":
        vlm_provider = OpenAIVisionProvider(api_key=vlm_api_key)

    # Initialize PDF parser
    parser = PDFParser(llm_provider=llm_provider,
                       vlm_provider=vlm_provider, process_sequentially=True, ocr_fallback=True)

    # Parse the PDF
    result = parser.parse_pdf(pdf_url)

    # Save output to file if needed
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)  # Save with pretty formatting

    return result
