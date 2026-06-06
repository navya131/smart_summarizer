from flask import Flask, render_template, request

from summarizer import summarize_text, calculate_similarity

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    original_text = ""
    summarized_text = ""

    fuzzy_score = ""
    cosine_score = ""
    final_score = ""

    if request.method == "POST":

        original_text = request.form["text"]

        # Generate Summary
        summarized_text = summarize_text(original_text)

        # Similarity Scores
        scores = calculate_similarity(
            original_text,
            summarized_text
        )

        fuzzy_score = scores["fuzzy_score"]
        cosine_score = scores["cosine_score"]
        final_score = scores["final_score"]

    return render_template(
        "index.html",
        original_text=original_text,
        summarized_text=summarized_text,
        fuzzy_score=fuzzy_score,
        cosine_score=cosine_score,
        final_score=final_score
    )

if __name__ == "__main__":
    app.run(debug=True)

