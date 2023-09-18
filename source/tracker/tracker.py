from os.path import exists


class Tracker():
    def __init__(self, infile):
        """
        Purpose: file
        """

        self.file = infile
        self.values = set(self.load())

    def load(self):
        if not exists(self.file):
            print(f"Creating {self.file}")
            open(self.file, "a").write("")

        f = open(self.file, 'r').read()
        return f.split("\n")

    def exists(self, value):
        return value in self.values

    def add(self, values):
        if not type(values) == type([]):
            values = [values]

        for val in values:
            self.values.add(val)

    def save(self):
        open(self.file, 'w').write("\n".join(sorted(self.values)) + "\n")
        return self
