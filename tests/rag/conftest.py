import pymupdf
import pytest


@pytest.fixture
def make_pdf(tmp_path):
    def factory(name, pages, toc=None):
        path = tmp_path / name
        doc = pymupdf.open()
        for blocks in pages:
            page = doc.new_page(width=595, height=842)
            for y, text in blocks:
                page.insert_text((72, y), text, fontname="korea", fontsize=11)
        if toc:
            doc.set_toc(toc)
        doc.save(path)
        doc.close()
        return path

    return factory
