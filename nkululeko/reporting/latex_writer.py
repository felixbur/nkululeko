"""
latex_writer.py
print out report as latex file and pdf
"""

from pylatex import Document, Section, Subsection, Command, Figure
from pylatex.utils import italic, NoEscape
from nkululeko.reporting.report_item import ReportItem
from nkululeko.util import Util


class LatexWriter:
    def __init__(self):
        self.util = Util("LatexWriter")
        title = self.util.config_val("REPORT", "title", "Nkululeko report")
        author = self.util.config_val("REPORT", "author", "anon")
        doc = Document()
        doc.preamble.append(Command("title", title))
        doc.preamble.append(Command("author", author))
        doc.preamble.append(Command("date", NoEscape(r"\today")))
        doc.append(NoEscape(r"\maketitle"))
        self.doc = doc

    def add_items_for_section(self, section, contents):
        with self.doc.create(Section(section)):
            for ri in contents:
                with self.doc.create(Subsection(ri.caption)):
                    self.doc.append(ri.contents)
                    if ri.has_image:
                        with self.doc.create(Figure(position='h!')) as pic:
                            pic.add_image(ri.image, width='250px')
                            pic.add_caption(ri.caption)

    def finish_doc(self):
        target_filename = self.util.config_val(
            "REPORT", "latex", "nkululeko_latex"
        )
        self.doc.generate_pdf(target_filename, clean_tex=False)
