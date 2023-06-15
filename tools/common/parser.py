import sys
import argparse


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        print('\nerror: {0}'.format(message))
        sys.exit(1)
