#!/bin/bash

set -ex

source $(dirname $0)/activate someenv

exec tfi "$@"
