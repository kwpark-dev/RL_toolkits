.def defines configuration of linux os. Check apptainer documentation. Install kinova pkgs manually.

1. sudo apptainer build --sandbox ubuntu_full_sandbox_kinova ubuntu_full.def
2. sudo apptainer shell --no-home --writable ubuntu_full_sandbox_kinova

Once you want to run example files for python api, please add 

import collections.abc
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSequence = collections.abc.MutableSequence

to top of .py file so that you avoid version-mismatch error.
