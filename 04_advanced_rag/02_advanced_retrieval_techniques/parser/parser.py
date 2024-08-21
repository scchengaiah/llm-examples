from parser.pdf.pdf_parser import PDFParser


def parse(file_obj, describe_images=False):
    if file_obj.type in ['application/pdf', 'pdf']:
        pdf_parser = PDFParser(file_obj, describe_images=describe_images)
        return pdf_parser.parse()
    else:
        raise NotImplementedError("Currently only PDF files are supported")
