#!/bin/bash

# Create lcov report
# capture coverage info
lcov --directory . --capture --output-file coverage.info
# filter out system and extra files.
# To also not include test code in coverage add them with full path to the patterns: '*/tests/*'
lcov --remove coverage.info '/usr/*' '*Developer/CommandLineTools/*' '*/root/*' '*-install/*' --output-file coverage.info
# output coverage data for debugging (optional)
lcov --list coverage.info
# Uploading to CodeCov
# '-f' specifies file(s) to use and disables manual coverage gathering and file search which has already been done above
bash <(curl -s https://codecov.io/bash) -v -d -f coverage.info || echo "Codecov did not collect coverage reports"
bash <(curl -s https://codecov.io/bash) -v -f coverage.info || echo "Codecov did not collect coverage reports"
