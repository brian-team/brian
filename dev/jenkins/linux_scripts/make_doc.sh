#!/bin/bash -x

# Use the source directory
export PYTHONPATH="$(pwd)"

# Generates the HTML and PDF documentation for brian
cd dev/tools/docs

# Generate HTML documentation
python generate_html.py || exit 1

# Generate PDF documentation
python build_latex.py || exit 1

cd ../../..

# Copy the PDF to the doc directory
cp "$DOCROOT"/docs_sphinx/_latexbuild/Brian*.pdf "$DOCROOT"/docs/

# Delete old docs zip file if it exists
rm docs.zip 2> /dev/null || :

# copy the docs to the directory served by the local webserver
cp -r "$DOCROOT"/docs/* ~/www-doc/"$DOCNAME"

# Create new zip file
cd "$DOCROOT"
zip -r -q -9 docs.zip docs/
