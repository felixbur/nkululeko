"""
report.py

Collector class for report items collected during module processing.

"""

from nkululeko.utils.util import Util


class Report:
    def __init__(self):
        self.report_items = {}
        self.util = Util("Report")
        self.initial = True

    def add_item(self, ri):
        if ri.topic in self.report_items:
            self.report_items[ri.topic].append(ri)
        else:
            self.report_items[ri.topic] = []
            self.report_items[ri.topic].append(ri)

    def print(self):
        print("###### Nkululeko Report ######")
        for topic in self.report_items:
            print(f"### {topic}")
            for c in self.report_items[topic]:
                print(c.caption)
                print("\t" + c.contents)

    def export_latex(self):
        if str(self.util.config_val("REPORT", "show", "False")).lower() == "true":
            from nkululeko.reporting.latex_writer import LatexWriter

            lw = LatexWriter()
            for topic in self.report_items:
                lw.add_items_for_section(topic, self.report_items[topic])
            lw.finish_doc()
