# Generates the HTML and PDF documentation for brian
cd dev/tools/docs

# Generate HTML documentation
python generate_html.py || exit 1

# Generate PDF documentation
python build_latex.py || exit 1

cd ../../..

# Copy the PDF to the doc directory
cp docs_sphinx/_latexbuild/Brian.pdf docs/

# Delete old docs zip file if it exists
rm docs.zip 2> /dev/null || :

# Create new zip file
zip -r -q -9 docs.zip docs/