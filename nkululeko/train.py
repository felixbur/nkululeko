#!/usr/bin/env python3

# train.py: Entry script to do a Nkululeko experiment (alias for nkululeko.py)

from nkululeko.nkululeko import doit, main  # noqa: F401

if __name__ == "__main__":
    main()  # use this if you want to state the config file path on command line
