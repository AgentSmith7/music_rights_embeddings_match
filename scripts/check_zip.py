#!/usr/bin/env python3
import zipfile
import sys

z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
print(f"Total files: {len(names)}")
for n in names[:100]:
    print(n)
