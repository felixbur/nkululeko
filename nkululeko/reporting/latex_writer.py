"""
latex_writer.py
print out report as latex file and pdf
"""

from pylatex import Command, Document, Figure, Section, Subsection
from pylatex.utils import NoEscape

from nkululeko.utils.util import Util


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
                        with self.doc.create(Figure(position="h!")) as pic:
                            pic.add_image(ri.image, width="250px")
                            pic.add_caption(ri.caption)
                            # reference = pic.dumps_as_content()
                            # self.doc.append(f"See figure: {reference}")

    def finish_doc(self):
        from subprocess import CalledProcessError

        target_filename = self.util.config_val("REPORT", "latex", "nkululeko_latex")
        target_dir = self.util.get_exp_dir()
        path = "/".join([target_dir, target_filename])
        self.util.debug(f"Generated latex report to {path}")
        try:
            self.doc.generate_pdf(path, clean_tex=False)
        except CalledProcessError as e:
            self.util.debug(f"Error while generating PDF file: {e}")
