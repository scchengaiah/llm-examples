## Docling parses documents and exports them to the desired format with ease and speed.
## https://github.com/DS4SD/docling
## https://ds4sd.github.io/docling/

## Technical Paper: https://arxiv.org/pdf/2408.09869

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.tesseract_ocr_cli_model import TesseractCliOcrOptions
from docling.models.tesseract_ocr_model import TesseractOcrOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
import os
import logging
import time
import json
from pathlib import Path


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
image_dir = os.path.join(data_dir, "images")
temp_dir = os.path.join(project_root, "temp")

# Create dirs.
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# pdf_file_path = os.path.join(data_dir, "jio-financial-services-annual-report-2023-2024-small.pdf")

# pdf with images and tables (Docling Technical paper)
pdf_file_path = "https://arxiv.org/pdf/2408.09869"

_log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0

def simple_example():
    print("Instantiating DocumentConverter...")
    converter = DocumentConverter()

    start_time = time.time()

    print("Conversion Started...")
    result = converter.convert(pdf_file_path)
    print("Exporting to Markdown...")
    print(result.document.export_to_markdown())

    print(f"Time taken for conversion: {time.time() - start_time}")

# simple_example()

## https://ds4sd.github.io/docling/examples/custom_convert/
def custom_example():
    logging.basicConfig(level=logging.INFO)

    input_doc_path = Path("./tests/data/2206.01062.pdf")

    ###########################################################################

    # The following sections contain a combination of PipelineOptions
    # and PDF Backends for various configurations.
    # Uncomment one section at the time to see the differences in the output.

    # -----------------------------------------------------------------------------------------------
    # PyPdfium without EasyOCR
    # -----------------------------------------------------------------------------------------------

    # OBSERVATION:
    # 1. Faster conversion. 21 pages converted in 13 seconds on GPU.
    # 2. The markdown output looks good without much distractions for the pdf data/jio-financial-services-annual-report-2023-2024-small.pdf
    # 3. On passing the entire markdown content for the above pdf, we were able to get accurate response.
    # 4. It should be noted that there are no reference to images mentioned in the markdown output, also the table structure options are minimal.
    # 5. PyPdfium is a faster parser compared to docling parser. Docling parser is claimed to be more effective and has low level features implemented.
    #    Refer to the technical paper mentioned in the above comments for more details.

    # CODE:

    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = False
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = False

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(
    #             pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
    #         )
    #     }
    # )

    # -----------------------------------------------------------------------------------------------
    # PyPdfium with EasyOCR
    # -----------------------------------------------------------------------------------------------

    # OBSERVATION:
    # 1. Took 50 seconds for the pdf data/jio-financial-services-annual-report-2023-2024-small.pdf. 
    # 2. The markdown output looks similar to the test without EasyOCR.
    # 3. On passing the entire markdown content for the above pdf, we were able to get accurate response.
    # 4. Able to recognize complex table structures.

    # CODE:

    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use more accurate TableFormer model
    # 
    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(
    #             pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
    #         )
    #     }
    # )

    # -----------------------------------------------------------------------------------------------
    # Docling Parse without EasyOCR
    # -----------------------------------------------------------------------------------------------

    # OBSERVATION:
    # 1. As per the technical paper, docling parser is more efficient compared to PyPdfiumDocumentBackend but can be slightly slower.
    # 2. The markdown output looks slightly better, but there is no major deviation atleast for the above PDF files.
    # 3. On passing the entire markdown content for the above pdf, we were able to get accurate response.
    # 4. Able to recognize complex table structures.
    # 5. Took 42 seconds for the pdf data/jio-financial-services-annual-report-2023-2024-small.pdf.

    # CODE:

    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = False
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # -----------------------------------------------------------------------------------------------
    # Docling Parse with EasyOCR
    # -----------------------------------------------------------------------------------------------

    # OBSERVATION:
    # 1. As per the technical paper, docling parser is more efficient compared to PyPdfiumDocumentBackend but can be slightly slower.
    # 2. The markdown output looks slightly better, but there is no major deviation atleast for the above PDF files.
    # 3. On passing the entire markdown content for the above pdf, we were able to get accurate response.
    # 4. Able to recognize complex table structures.
    # 5. Took 80 seconds for the pdf data/jio-financial-services-annual-report-2023-2024-small.pdf.
    # 6. Could not find major quality differences with respect to the above examples.

    # CODE:

    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use more accurate TableFormer model

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # -----------------------------------------------------------------------------------------------
    # Docling Parse with EasyOCR (CPU only)
    # -----------------------------------------------------------------------------------------------

    # OBSERVATION:
    # 1. Did not test this since the CPU variant takes more time and the output is similar.

    # CODE:

    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.ocr_options.use_gpu = False  # <-- set this.
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # -----------------------------------------------------------------------------------------------
    # Docling Parse with Tesseract
    # -----------------------------------------------------------------------------------------------

    # PRE-REQUISITES:
    # Install Tesseract by following the documentation here - https://ds4sd.github.io/docling/installation/#development-setup
    # STEPS: 
    # apt-get install tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev pkg-config
    # TESSDATA_PREFIX=$(dpkg -L tesseract-ocr-eng | grep tessdata$)
    # echo "Set TESSDATA_PREFIX=${TESSDATA_PREFIX}"

    # OBSERVATION:
    # 1. Took 68 seconds for jio-financial-services-annual-report-2023-2024-small.pdf
    # 2. Did not find a major difference atleast for the above pdf tested.

    # CODE:

    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options = TesseractOcrOptions()

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    # -----------------------------------------------------------------------------------------------
    # Docling Parse with Tesseract CLI
    # -----------------------------------------------------------------------------------------------

    # PRE-REQUISITES:
    # Install Tesseract by following the documentation here - https://ds4sd.github.io/docling/installation/#development-setup
    # STEPS: 
    # apt-get install tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev pkg-config
    # TESSDATA_PREFIX=$(dpkg -L tesseract-ocr-eng | grep tessdata$)
    # echo "Set TESSDATA_PREFIX=${TESSDATA_PREFIX}"

    # OBSERVATION:
    # 1. Took 68 seconds for jio-financial-services-annual-report-2023-2024-small.pdf
    # 2. Did not find a major difference atleast for the above pdf tested.

    # CODE:

    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options = TesseractCliOcrOptions()

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    ###########################################################################

    start_time = time.time()
    conv_result = doc_converter.convert(pdf_file_path)
    end_time = time.time() - start_time

    _log.info(f"Document converted in {end_time:.2f} seconds.")

    ## Export results
    output_dir = Path(temp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_result.input.file.stem

    # Export Deep Search document JSON format:
    with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(conv_result.document.export_to_dict()))

    # Export Text format:
    with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_text())

    # Export Markdown format:
    with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_markdown())

    # Export Document Tags format:
    with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_document_tokens())

# Refer to the detailed comments inside each aspect of the implementation to understand better.
# Uncomment the required sections to test and comment the other sections.
# Each section is divided based on this delimiter
# # -----------------------------------------------------------------------------------------------
# custom_example()

def image_export():
    logging.basicConfig(level=logging.INFO)

    output_dir = Path(temp_dir)

    # Important: For operating with page images, we must keep them, otherwise the DocumentConverter
    # will destroy them for cleaning up memory.
    # This is done by setting PdfPipelineOptions.images_scale, which also defines the scale of images.
    # scale=1 correspond of a standard 72 DPI image
    # The PdfPipelineOptions.generate_* are the selectors for the document elements which will be enriched
    # with the image field
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = False
    pipeline_options.generate_table_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(pdf_file_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    # Save page images
    if pipeline_options.generate_page_images:
        for page_no, page in conv_res.document.pages.items():
            page_no = page.page_no
            page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
            with page_image_filename.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")

    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.image.pil_image.save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.image.pil_image.save(fp, "PNG")

    # Save markdown with embedded pictures
    # image_mode=ImageRefMode.EMBEDDED - Embeds base64 representation of the image in the markdown document.
    content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    with md_filename.open("w") as fp:
        fp.write(content_md)

    end_time = time.time() - start_time

    _log.info(f"Document converted and figures exported in {end_time:.2f} seconds.")

# image_export()

