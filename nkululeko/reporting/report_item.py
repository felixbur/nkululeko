""" reportitem.py
a basic report snippet.
"""

import os.path

class ReportItem:
    def __init__(self, topic, caption, contents, image=None):
        self.topic = topic
        self.caption = caption
        self.contents = contents
        self.has_image = False
        if image is not None:
            self.image = os.path.abspath(image)
            self.has_image = True

    def to_string(self):
        return (
            f"topic: {self.topic}, caption: {self.caption}, contents:"
            f" {self.contents}"
        )
