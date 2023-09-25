""" reportitem.py
a basic report snippet.
"""
class ReportItem:
    def __init__(self, topic, caption, contents):
        self.topic = topic
        self.caption = caption 
        self.contents = contents
        self.picture = None

    def add_picture(self, pic):
        self.picture = pic

    def to_string(self):
        return f'topic: {self.topic}, caption: {self.caption}, contents: {self.contents}'