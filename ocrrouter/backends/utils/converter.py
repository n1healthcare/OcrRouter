"""Converter utilities for transforming model outputs to middle JSON format."""

from loguru import logger

from ocrrouter.processing.magic_model import MagicModel
from ocrrouter.processing.utils.cut_image import cut_image_and_table
from ocrrouter.postprocessor.utils.table_merge import merge_table
from ocrrouter.utils.enum_class import ContentType
from ocrrouter.utils.hash_utils import bytes_md5
from ocrrouter.version import __version__


def blocks_to_page_info(
    page_blocks, image_dict, page, image_writer, page_index
) -> dict:
    """Convert blocks to page information dictionary.

    Args:
        page_blocks: List of ContentBlock objects for this page.
        image_dict: Dictionary containing page image and metadata.
        page: PyMuPDF page object.
        image_writer: Writer for saving extracted images.
        page_index: Zero-based page index.

    Returns:
        Dictionary containing page information with para_blocks, discarded_blocks, etc.
    """
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    width, height = map(int, page.get_size())

    magic_model = MagicModel(page_blocks, width, height)
    image_blocks = magic_model.get_image_blocks()
    table_blocks = magic_model.get_table_blocks()
    title_blocks = magic_model.get_title_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    code_blocks = magic_model.get_code_blocks()
    ref_text_blocks = magic_model.get_ref_text_blocks()
    phonetic_blocks = magic_model.get_phonetic_blocks()
    list_blocks = magic_model.get_list_blocks()

    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    all_spans = magic_model.get_all_spans()
    # Cut images for image/table/interline_equation spans
    for span in all_spans:
        if span["type"] in [
            ContentType.IMAGE,
            ContentType.TABLE,
            ContentType.INTERLINE_EQUATION,
        ]:
            span = cut_image_and_table(
                span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale
            )

    page_blocks = []
    page_blocks.extend(
        [
            *image_blocks,
            *table_blocks,
            *code_blocks,
            *ref_text_blocks,
            *phonetic_blocks,
            *title_blocks,
            *text_blocks,
            *interline_equation_blocks,
            *list_blocks,
        ]
    )
    # Sort page_blocks by index
    page_blocks.sort(key=lambda x: x["index"])

    page_info = {
        "para_blocks": page_blocks,
        "discarded_blocks": discarded_blocks,
        "page_size": [width, height],
        "page_idx": page_index,
    }
    return page_info


def result_to_middle_json(
    model_output_blocks_list,
    images_list,
    pdf_doc,
    image_writer,
    formula_enable: bool = True,
    table_enable: bool = True,
    table_merge_enable: bool = True,
):
    """Convert model output to middle JSON format.

    Args:
        model_output_blocks_list: List of page blocks (list[list[ContentBlock]]).
        images_list: List of image dictionaries with page images and metadata.
        pdf_doc: PyMuPDF document object.
        image_writer: Writer for saving extracted images.
        formula_enable: Whether formula extraction is enabled.
        table_enable: Whether table extraction is enabled.
        table_merge_enable: Whether cross-page table merging is enabled.

    Returns:
        Middle JSON dictionary containing pdf_info and metadata.
    """

    middle_json = {"pdf_info": [], "_backend": "vlm", "_version_name": __version__}
    for index, page_blocks in enumerate(model_output_blocks_list):
        page = pdf_doc[index]
        image_dict = images_list[index]
        page_info = blocks_to_page_info(
            page_blocks, image_dict, page, image_writer, index
        )
        middle_json["pdf_info"].append(page_info)

    # Cross-page table merge
    if table_merge_enable:
        merge_table(middle_json["pdf_info"])

    # Close pdf document
    pdf_doc.close()
    return middle_json
