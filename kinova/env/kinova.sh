#!/bin/sh

export APPTAINERENV_DISPLAY=$DISPLAY

xhost + > /dev/null 2>&1

sudo apptainer shell --no-home --writable ubuntu_full_sandbox_kinova


