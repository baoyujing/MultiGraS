import logging

from preprocessors.utils import hashhex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reader:
    def read_story(self, path):
        """
        Read .story file.
        """
        f = open(path)
        story = f.read()
        f.close()
        return story

    def get_split(self, path):
        split_list = []
        with open(path, "r") as f:
            for line in f:
                s = line.strip().encode("utf-8")
                hashid = hashhex(s)
                split_list.append(hashid)
        return split_list
