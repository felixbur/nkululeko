from pylatex import Document, Section, Subsection, Command
from pylatex.utils import italic, NoEscape
from nkululeko.reporting.report_item import ReportItem
from nkululeko.util import Util

class LatexWriter:
    def __init__(self):
        self.util = Util('LatexWriter')
        doc = Document()
        doc.preamble.append(Command('title', 'Nkululeko report'))
        doc.preamble.append(Command('author', 'anon'))
        doc.preamble.append(Command('date', NoEscape(r'\today')))
        doc.append(NoEscape(r'\maketitle'))
        self.doc = doc 

    def add_items_for_section(self, section, contents):
        with self.doc.create(Section(section)):
            for ri in contents:
                with self.doc.create(Subsection(ri.caption)):
                    self.doc.append(ri.contents)

    def finish_doc(self):
        self.doc.generate_pdf('nkululeko_latex', clean_tex=False)