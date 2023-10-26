import os
from flask import Flask, request, render_template, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import difflib

app = Flask(__name__)

document_folder = 'docs/'
student_files = [doc for doc in os.listdir(document_folder) if doc.endswith('.txt')]
student_notes = [open(os.path.join(document_folder, _file), encoding='utf-8').read() for _file in student_files]

def vectorize(Text):
    return TfidfVectorizer().fit_transform(Text).toarray()

def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])

vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()

# Define the HTML template as a multiline string
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Plagiarism Checker</title>
</head>
<body>
    <h1>Plagiarism Checker</h1>
    
    <form action="/check_plagiarism" method="post" enctype="multipart/form-data">
        <label for="file1">Select the first text document:</label>
        <input type="file" name="file1" accept=".txt" required><br><br>

        <label for="file2">Select the second text document:</label>
        <input type="file" name="file2" accept=".txt" required><br><br>

        <input type="submit" value="Check for Plagiarism">
    </form>
</body>
</html>
"""

def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results


@app.route('/check_plagiarism', methods=['GET', 'POST'])
def check_plagiarism_route():
    if request.method == 'POST':
        plagiarism_results = check_plagiarism()

        # Generate a PDF report with color-coded plagiarism and citations
        output_pdf = 'plagiarism_report.pdf'
        c = canvas.Canvas(output_pdf, pagesize=letter)
        
        for data in plagiarism_results:
            doc1, doc2, sim_score = data
            c.drawString(100, 700, f"Plagiarism detected between {doc1} and {doc2}")
            c.drawString(100, 680, f"Similarity Score: {sim_score:.2f}")

            # Get the content of the two documents
            content_doc1 = open(os.path.join(document_folder, doc1), encoding='utf-8').read()
            content_doc2 = open(os.path.join(document_folder, doc2), encoding='utf-8').read()

            # Use difflib to generate side-by-side comparison
            d = difflib.Differ()
            diff = list(d.compare(content_doc1.splitlines(), content_doc2.splitlines()))

            # Dynamically calculate y-position
            y_position = 660
            for line in diff:
                if line.startswith('  '):
                    c.drawString(100, y_position, line)
                    print("  ")
                else:
                    c.drawString(100, y_position, f'[{line[0]}]{line[2:]}')
                    print("  ")
                y_position -= 15  # Adjust this value as needed

        c.save()

        return send_file(output_pdf, as_attachment=True)

    return html_template  # Render the embedded HTML template


if __name__ == '__main__':
    app.run(debug=True, port=5000)
